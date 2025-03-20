## Getting started

I'm using `uv` to manage the python environment. It's like `pip` but much faster and better at managing dependencies. I can really recommend looking into this a bit more if you don't know it: https://astral.sh/uv/

```bash
# install uv for the shell 
curl -LsSf https://astral.sh/uv/install.sh | less

# OR you can use your current python environment
pip install uv
```

To install all the required packages and dependencies, run the following command in the project directory.
```bash
cd oceanco2-dataset-exploration  # the project directory
uv sync  # installs all required packages and dependencies
```

