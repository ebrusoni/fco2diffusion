import os
import polars as pl
from loguru import logger
import click  # Add click import


@click.command()
@click.argument("dest_dir")
@click.option("--version", default="v2024", help="SOCAT version to download.")
@click.option("--time_freq", default="1d", help="Time frequency for regridding.")
@click.option(
    "--bin_size_deg",
    default=0.05,
    type=float,
    help="Bin size in degrees for regridding.",
)
def cli(dest_dir, version, time_freq, bin_size_deg):
    """
    Command-line interface for SOCAT data processing.

    This function takes command-line arguments for the destination directory,
    SOCAT version, time frequency, and bin size for regridding. It then calls
    the main function to perform the data processing.
    """
    # Gather arguments and pass to main
    regrid_kwargs = {"time_freq": time_freq, "bin_size_deg": bin_size_deg}
    main(dest_dir, version, regrid_kwargs)


def main(
    dest_dir: str,
    version="v2024",
    regrid_kwargs=dict(time_freq="1d", bin_size_deg=0.05),
):
    """
    Downloads SOCAT data, processes it to parquet format, and regrids the data.

    The regridding process involves binning the data according to the specified time frequency
    and spatial bin size. The processed data is then saved as a parquet file.

    Parameters
    ----------
    dest_dir : str
        Directory to save the output files.
    version : str, optional
        Version of SOCAT data to download. Defaults to "v2024".
    regrid_kwargs : dict, optional
        Dictionary of regridding parameters, including 'time_freq' and 'bin_size_deg'.
        Defaults to binning at a daily frequency ('1d') and a spatial resolution of 0.05 degrees.

    Returns
    -------
    None
        The function saves the processed data as a parquet file in the specified directory.
    """
    # Prepare the SOCAT data URL
    url = f"https://socat.info/socat_files/{version}/SOCAT{version}.tsv.zip"
    # Convert the SOCAT TSV file to parquet format
    fname_pq = socat_tsv_to_parquet(dest_dir, url=url)[1]

    # Read the parquet file into a Polars DataFrame
    df = pl.read_parquet(fname_pq)
    # Regrid the SOCAT data
    df = regrid_socat_data(
        df, lon_col="longitude_dec_deg_e", lat_col="latitude_dec_deg_n", **regrid_kwargs
    )

    # Construct the output filename based on regridding parameters
    time_str = regrid_kwargs["time_freq"]
    res_str = str(regrid_kwargs["bin_size_deg"]).replace(".", "")
    # Save the regridded data to a parquet file
    df.write_parquet(f"{dest_dir}/SOCAT{version}_{time_str}_{res_str}deg.pq")


def socat_tsv_to_parquet(
    dest_dir: str,
    url: str = "https://socat.info/socat_files/v2024/SOCATv2024.tsv.zip",
    **kwargs,
) -> tuple[str, str]:
    """
    Retrieves and processes the SOCAT TSV file from a URL, then saves it in Parquet format.

    This function downloads the SOCAT data, unzips it, reads the TSV file,
    and saves it as a Parquet file for faster access and efficient storage.

    Parameters
    ----------
    dest_dir : str
        Target directory for storing retrieved files.
    url : str, optional
        SOCAT data URL. Defaults to the v2024 version.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    tuple of str
        A tuple containing the downloaded filename and the output parquet filename.
    """
    # Download the file and handle any existing parquet
    import pooch

    # Use pooch to download the file, unzip it, and return the filename
    fname = pooch.retrieve(
        url,
        known_hash=None,
        fname=url.split("/")[-1],
        path=dest_dir,
        progressbar=True,
        processor=pooch.Unzip(),
    )[0]

    # Define the name for the output parquet file
    name = url.split("/")[-1].split(".")[0]
    sname = os.path.join(dest_dir, f"{name}.pq")
    # Check if the parquet file already exists
    if os.path.isfile(sname):
        logger.info(f"Returning existing file: {sname}")
        return fname, sname
    else:
        logger.info("Processing raw SOCAT data")

    # Determine the starting line of the data in the TSV file
    starting_line = _get_line_num_match_start_end(
        fname, line_start="Expocode", line_end="fCO2rec_flag"
    )
    logger.debug(f"Starting line of {fname}: {starting_line}")

    # Read the TSV file into a Polars DataFrame
    logger.info(f"Reading in file: {fname}")
    df = pl.read_csv(
        fname,
        skip_rows=starting_line - 2,
        skip_rows_after_header=2,
        separator="\t",
        columns=[  # some preselected columns
            "Expocode",
            "yr",
            "mon",
            "day",
            "hh",
            "mm",
            "ss",
            "longitude [dec.deg.E]",
            "latitude [dec.deg.N]",
            "dist_to_land [km]",
            "sal",
            "WOA_SSS",
            "SST [deg.C]",
            "PPPP [hPa]",
            "NCEP_SLP [hPa]",
            "fCO2rec [uatm]",
            "fCO2rec_flag",
        ],
        dtypes={  # read these as strings to make date conversion easier
            "yr": str,
            "mon": str,
            "day": str,
            "hh": str,
            "mm": str,
            "ss": str,
            "Expocode": str,
            "dist_to_land [km]": float,
        },
        infer_schema_length=1000,
    )

    # Rename the columns to a standardized format
    df = df.rename({s: _make_col_name_compat(s) for s in df.columns})

    # Convert the date and time strings to a datetime64 column
    df = _convert_datestr_to_time(df)

    # Save the DataFrame to a parquet file
    logger.info(f"Saving data to parquet: {sname}")
    df.write_parquet(sname)

    return fname, sname


def _convert_datestr_to_time(
    df: pl.DataFrame,
    date_cols=["yr", "mon", "day", "hh", "mm", "ss"],
    format="%Y%m%d%H%M%S.",
) -> pl.DataFrame:
    """
    Converts date/time columns to a single datetime64 column in a Polars DataFrame.

    This function concatenates the specified date and time columns into a single
    string, then converts that string to a datetime64 data type. The original
    date and time columns are then dropped from the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing date/time columns.
    date_cols : list of str, optional
        List of date column names to concatenate.
    format : str, optional
        Datetime format string.

    Returns
    -------
    pl.DataFrame
        DataFrame with a new 'time' column replacing date/time columns.
    """
    # Build a 'time' column by concatenating date/time strings
    logger.info("Converting date and time strings to datetime64")
    df = (
        df.with_columns(pl.concat_str(date_cols, separator="").alias("time"))
        .with_columns(
            pl.col("time").str.strptime(pl.Datetime, format=format).alias("time")
        )
        .drop(*date_cols)
    )

    return df


def _get_line_num_match_start_end(
    fname: str, line_start: str = "Expocode", line_end: str = "fCO2rec_flag"
) -> int:
    """
    Finds the line number in a file where data starts, matching both start & end markers.

    This function reads a file line by line, searching for a line that starts with
    the specified `line_start` and ends with the specified `line_end`. The line number
    of the matching line is then returned.

    Parameters
    ----------
    fname : str
        Filename to read from.
    line_start : str, optional
        Start marker. Defaults to "Expocode".
    line_end : str, optional
        End marker. Defaults to "fCO2rec_flag".

    Returns
    -------
    int
        Line number where data begins.
    """
    # Look for lines matching start and end criteria
    line_end = line_end + "\n"
    with open(fname, encoding="latin") as tsv:
        start_line = 0
        # Iterate through each line in the file
        for n, line in enumerate(tsv):
            if start_line:
                break
            # Check if the line starts with line_start and ends with line_end
            if line.startswith(line_start) & line.endswith(line_end):
                start_line = n

    return start_line


def _make_col_name_compat(s):
    """
    Converts column names to a standardized format (lowercase with underscores).

    This function takes a column name as input and converts it to a standardized
    format by replacing non-alphanumeric characters with underscores, removing
    duplicate underscores, and converting the name to lowercase.

    Parameters
    ----------
    s : str
        Original column name.

    Returns
    -------
    str
        Converted column name (e.g., 'some_column').
    """
    # Use a regex to replace invalid characters
    import re

    # Define a regular expression pattern to match non-alphanumeric characters
    ptn = re.compile(r"\W+")
    # Replace non-alphanumeric characters with underscores
    s = ptn.sub("_", s)
    # Replace duplicate underscores with a single underscore
    s = s.replace("__", "_")
    # Remove trailing underscores
    s = s[:-1] if s.endswith("_") else s
    # Convert the name to lowercase
    s = s.lower()
    return s


def regrid_socat_data(
    df: pl.DataFrame,
    bin_size_deg=0.05,
    time_freq="1d",
    time_col="time",
    lat_col="lat",
    lon_col="lat",
):
    """
    Regrids the SOCAT data by rounding location and time columns, then aggregating.

    This function regrids the SOCAT data by binning the latitude, longitude, and
    time columns. The data is binned by rounding the latitude and longitude to the
    nearest `bin_size_deg` degrees, and rounding the time to the nearest `time_freq`.
    The data is then aggregated by the binned latitude, longitude, time, and expocode.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    bin_size_deg : float, optional
        Spatial bin size in degrees. Defaults to 0.05.
    time_freq : str, optional
        Temporal bin size (e.g., '1d' for daily). Defaults to '1d'.
    time_col : str, optional
        Column name for time data. Defaults to "time".
    lat_col : str, optional
        Column name for latitude data. Defaults to "lat".
    lon_col : str, optional
        Column name for longitude data. Defaults to "lat".

    Returns
    -------
    pl.DataFrame
        Regridded Polars DataFrame with aggregated columns.
    """

    # Helper function to bin data according to provided bin sizes
    def bin_expr(col_name, bin_size, lower, upper):
        # Calculate center-based bins and clamp values within specified range
        half_bin = bin_size / 2
        offset = lower + half_bin  # bin centers start at lower + half_bin
        return (
            ((pl.col(col_name) - offset) / bin_size + 0.5).floor() * bin_size + offset
        ).clip(offset, upper - half_bin)

    logger.info(f"Regridding SOCAT data to {bin_size_deg} degree bins")
    res = str(bin_size_deg).replace(".", "")
    df = (
        df.with_columns(
            [
                pl.col(time_col).dt.round(time_freq).alias(f"time_{time_freq}"),
                bin_expr(lon_col, bin_size_deg, 0, 360).alias(f"lon_{res}"),
                bin_expr(lat_col, bin_size_deg, -90, 90).alias(f"lat_{res}"),
            ]
        )
        .group_by([f"time_{time_freq}", f"lat_{res}", f"lon_{res}", "expocode"])
        .agg(
            pl.mean(time_col).alias("time_avg"),
            pl.mean("sst_deg_c"),
            pl.mean("sal").cast(pl.Float32),
            pl.mean("pppp_hpa").cast(pl.Float32),
            pl.mean("woa_sss").cast(pl.Float32),
            pl.mean("ncep_slp_hpa").cast(pl.Float32),
            pl.mean("dist_to_land_km").cast(pl.Float32),
            pl.mean("fco2rec_uatm"),
            pl.mean("fco2rec_flag").cast(pl.Int8),
            pl.count("fco2rec_uatm").alias("n_samples").cast(pl.UInt16),
        )
        .sort(["expocode", f"time_{time_freq}"])
    )

    return df


if __name__ == "__main__":
    cli()
