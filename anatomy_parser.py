import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABELS = {
    0: "O",
    1: "B-ANATOMY",
    2: "I-ANATOMY",
    3: "B-OBS",
    4: "I-OBS",
    5: "B-MOD",
    6: "I-MOD",
}
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

def extract_AOM(text):
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
    current_label, buffer = None, []

    for token, pred in zip(tokens, preds):
        label = LABELS[pred]

        if label.startswith("B-"):
            if current_label:
                entities[current_label].append(" ".join(buffer))
            current_label = label[2:]
            buffer = [token.replace("##", "")]
        
        elif label.startswith("I-") and current_label == label[2:]:
            buffer.append(token.replace("##", ""))
        
        else:
            if current_label:
                entities[current_label].append(" ".join(buffer))
            current_label, buffer = None, []

    if current_label:
        entities[current_label].append(" ".join(buffer))

    return entities


caption = "There is a small nodule in the right upper lobe of the lung."

entities = extract_AOM(caption)
print(entities)

{
 'ANATOMY': ['right upper lobe', 'lung'],
 'OBS': ['nodule'],
 'MOD': ['small', 'right']
}