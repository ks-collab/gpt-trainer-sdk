## Installation

We recommend installing using the GitHub HTTPS URL:

```bash
pip install gpt_trainer_sdk@git+https://github.com/ks-collab/gpt-trainer-sdk.git
```

Updates can be installed with:

```bash
pip install --upgrade gpt_trainer_sdk@git+https://github.com/ks-collab/gpt-trainer-sdk.git
```

## Usage

You will need a GPT-trainer API key, which can be generated in the "Account" menu.

Initialize the SDK with your API key:
```python
from gpt_trainer_sdk import GPTTrainer

gpt_trainer = GPTTrainer(api_key="GPT_TRAINER_API_KEY")
```

## Development

There are some tests that incur costs via API calls. To run these tests, use the `-m incur_costs` flag.
To run all tests, use the following commands:

```bash
uv run pytest 
uv run pytest -m incur_costs
```