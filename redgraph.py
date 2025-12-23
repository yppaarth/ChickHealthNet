pip install transformers torch

MODEL_NAME = "StanfordAIMI/RadGraph"

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

ID2LABEL = model.config.id2label

B-ANATOMY, I-ANATOMY
B-OBSERVATION, I-OBSERVATION
B-MODIFIER, I-MODIFIER

def extract_AOM_radgraph(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = {"ANATOMY": [], "OBS": [], "MOD": []}
    current_type, buffer = None, []

    for tok, pred in zip(tokens, preds):
        label = ID2LABEL[pred]

        if label.startswith("B-"):
            if current_type:
                entities[current_type].append(" ".join(buffer))
            ent = label[2:]

            if ent == "OBSERVATION":
                current_type = "OBS"
            elif ent == "MODIFIER":
                current_type = "MOD"
            elif ent == "ANATOMY":
                current_type = "ANATOMY"
            else:
                current_type = None

            buffer = [tok.replace("##", "")]

        elif label.startswith("I-") and current_type:
            buffer.append(tok.replace("##", ""))

        else:
            if current_type:
                entities[current_type].append(" ".join(buffer))
            current_type, buffer = None, []

    if current_type:
        entities[current_type].append(" ".join(buffer))

    return entities

caption = "There is a small nodule in the right upper lobe of the lung."

print(extract_AOM_radgraph(caption))

{
 'ANATOMY': ['right upper lobe', 'lung'],
 'OBS': ['nodule'],
 'MOD': ['small', 'right']
}