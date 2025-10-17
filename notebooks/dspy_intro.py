import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import dspy
    from rich import print


    lm = dspy.LM(
        "openai/hf.co/unsloth/Qwen3-4B-GGUF:BF16",
    )
    return dspy, lm, print


@app.cell
def _(lm):
    lm("Why is the sky blue ??")
    return


@app.cell
def _(lm, print):
    print(lm.history[0]["response"])
    return


@app.cell
def _(dspy, lm):
    from pydantic import BaseModel, Field
    from typing import List

    class Character(BaseModel):
        name: str 
        background: str = Field(description="One short sentence giving the background of the character.")
        age: int

    class CreatePersona(dspy.Signature):
        """Create 3 different character"""
        theme: str = dspy.InputField()
        characters: List[Character] = dspy.OutputField()

    dspy.configure(lm=lm)
    create_persona = dspy.Predict(CreatePersona)
    return (create_persona,)


@app.cell
def _(create_persona, print):
    print(create_persona(theme="Space marines"))
    return


@app.cell
def _():
    from datasets import Dataset, load_dataset
    return Dataset, load_dataset


@app.cell
def _(load_dataset):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    return (ds,)


@app.cell
def _(ds):
    ds
    return


@app.cell
def _(ds):
    ds[0]
    return


@app.cell
def _():
    from copy import deepcopy

    def add_wrong_hint(row: dict) -> dict:
        row = deepcopy(row)
        wrong_answers = [label for label in row["choices"]["label"] if label != row["answerKey"]]
        row["question"] = f"<answer>{wrong_answers[0]}</answer>\n{row['question']}"
        return row
    return (add_wrong_hint,)


@app.cell
def _(add_wrong_hint, ds):
    add_wrong_hint(ds[0])
    return


@app.cell
def _(Dataset, add_wrong_hint):
    def create_dataset_with_hints(ds: Dataset):
        return ds.map(add_wrong_hint)
    return (create_dataset_with_hints,)


@app.cell
def _(create_dataset_with_hints, ds):
    ds_with_hints = create_dataset_with_hints(ds)
    return


if __name__ == "__main__":
    app.run()
