"""
entity_recognizer.py
Rule-based medical NER — symptoms, diseases, treatments, medications, body parts.
"""

from dataclasses import dataclass, field
from typing import List, Dict

ENTITY_DICT = {
    "symptom": [
        "fever", "cough", "fatigue", "headache", "nausea", "vomiting",
        "dizziness", "pain", "swelling", "rash", "shortness of breath",
        "chest pain", "weight loss", "loss of appetite", "insomnia",
        "blurred vision", "numbness", "tingling", "weakness", "tremor",
        "seizure", "confusion", "memory loss", "diarrhea", "constipation",
        "itching", "bruising", "bleeding", "inflammation", "stiffness",
        "wheezing", "palpitations", "sweating", "chills", "sore throat",
        "runny nose", "muscle ache", "joint pain", "back pain",
    ],
    "disease": [
        "diabetes", "hypertension", "asthma", "cancer", "arthritis",
        "depression", "anxiety", "alzheimer", "dementia", "parkinson",
        "stroke", "heart disease", "heart attack", "pneumonia", "tuberculosis",
        "HIV", "AIDS", "hepatitis", "cirrhosis", "kidney disease",
        "liver disease", "obesity", "anemia", "hypothyroidism", "COPD",
        "COVID-19", "influenza", "sepsis", "osteoporosis", "eczema",
        "psoriasis", "celiac", "crohn", "colitis", "appendicitis",
        "lupus", "multiple sclerosis", "epilepsy", "migraine",
    ],
    "treatment": [
        "surgery", "chemotherapy", "radiation", "medication", "therapy",
        "vaccine", "antibiotic", "insulin", "dialysis", "transplant",
        "physical therapy", "immunotherapy", "psychotherapy", "counseling",
        "blood transfusion", "stem cell", "targeted therapy", "hormone therapy",
        "biopsy", "endoscopy", "MRI", "CT scan", "X-ray", "ultrasound",
        "ECG", "blood test", "urine test", "exercise", "diet",
    ],
    "medication": [
        "metformin", "aspirin", "ibuprofen", "acetaminophen", "paracetamol",
        "lisinopril", "atorvastatin", "omeprazole", "amoxicillin", "penicillin",
        "prednisone", "warfarin", "levothyroxine", "amlodipine", "metoprolol",
        "sertraline", "fluoxetine", "gabapentin", "albuterol", "donepezil",
        "memantine", "dexamethasone", "paxlovid", "remdesivir",
    ],
    "body_part": [
        "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine",
        "colon", "pancreas", "thyroid", "blood", "bone", "joint", "muscle",
        "skin", "eye", "ear", "nose", "throat", "spine", "nerve",
        "artery", "vein", "bladder", "prostate", "uterus",
    ],
}

ENTITY_COLORS = {
    "symptom":    "#fef3c7",
    "disease":    "#fee2e2",
    "treatment":  "#d1fae5",
    "medication": "#dbeafe",
    "body_part":  "#ede9fe",
}

ENTITY_ICONS = {
    "symptom":    "🤒",
    "disease":    "🦠",
    "treatment":  "💊",
    "medication": "💉",
    "body_part":  "🫀",
}


@dataclass
class RecognitionResult:
    entities: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> Dict[str, List[str]]:
        return {k: list(set(v)) for k, v in self.entities.items() if v}


def recognize(text: str) -> RecognitionResult:
    lower = text.lower()
    found: Dict[str, List[str]] = {k: [] for k in ENTITY_DICT}

    # Sort terms by length desc to match longer phrases first
    all_terms = sorted(
        [(term, label) for label, terms in ENTITY_DICT.items() for term in terms],
        key=lambda x: len(x[0]), reverse=True,
    )

    for term, label in all_terms:
        idx = lower.find(term)
        if idx == -1:
            continue
        end = idx + len(term)
        before_ok = idx == 0 or not lower[idx-1].isalpha()
        after_ok = end == len(lower) or not lower[end].isalpha()
        if before_ok and after_ok:
            found[label].append(text[idx:end])

    return RecognitionResult(entities=found)
