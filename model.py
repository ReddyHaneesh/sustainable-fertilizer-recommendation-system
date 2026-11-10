"""
Sustainable Fertilizer Recommendation System - Machine Learning Model
---------------------------------------------------------------------
This module trains a Decision Tree Classifier using soil nutrient levels (N, P, K),
soil pH, and crop type to recommend the most suitable and sustainable fertilizer.

Suitable for college demonstration and viva presentation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# File path constants
DATASET_PATH = os.path.join(os.path.dirname(__file__), "fertilizer_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fertilizer_model.pkl")

# Supported crops for easy reference
SUPPORTED_CROPS = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Sugarcane",
    "Tomato",
    "Potato",
    "Groundnut",
]

# Educational knowledge base for fertilizers
FERTILIZER_KNOWLEDGE_BASE = {
    "Urea": {
        "full_name": "Urea (46% Nitrogen)",
        "npk_ratio": "46-0-0",
        "reason": "Nitrogen levels are low while Phosphorus and Potassium are relatively adequate. Nitrogen is essential for leafy vegetative growth, chlorophyll production, and rapid plant development.",
        "soil_context": "Recommended when soil exhibits low Nitrogen (N < 30 kg/ha or ppm) under normal pH conditions.",
        "usage_guidance": "Apply in split doses (e.g., basal and vegetative stages) rather than all at once. Avoid applying before heavy rain to minimize nitrogen runoff.",
        "environmental_benefits": "Applying only when nitrogen is deficient prevents groundwater nitrate contamination and reduces nitrous oxide (a potent greenhouse gas) emissions.",
        "sustainable_tips": "Consider using Neem-coated urea to slow down release, and incorporate green manures or legumes into your crop rotation to rebuild natural soil nitrogen.",
        "type": "Chemical / Mineral",
    },
    "DAP": {
        "full_name": "DAP (Diammonium Phosphate, 18-46-0)",
        "npk_ratio": "18-46-0",
        "reason": "Phosphorus is deficient in the soil. Phosphorus is vital for vigorous root development, energy transfer (ATP), and early crop establishment.",
        "soil_context": "Recommended when soil Phosphorus is low (P < 20 kg/ha or ppm).",
        "usage_guidance": "Apply at the time of sowing or transplanting near the root zone (basal dressing) where young roots can immediately access it.",
        "environmental_benefits": "Precision localized placement prevents phosphorus fixation in the soil and reduces surface phosphate runoff that causes algal blooms in waterways.",
        "sustainable_tips": "Inoculate seeds with Phosphate Solubilizing Bacteria (PSB) or mycorrhizae to mobilize native insoluble phosphorus in the soil.",
        "type": "Chemical / Mineral",
    },
    "MOP": {
        "full_name": "MOP (Muriate of Potash, 0-0-60)",
        "npk_ratio": "0-0-60",
        "reason": "Potassium levels in the soil are depleted. Potassium regulates stomatal water opening, strengthens plant cell walls, enhances drought resistance, and prevents pest infestation.",
        "soil_context": "Recommended when soil Potassium is low (K < 20 kg/ha or ppm).",
        "usage_guidance": "Best applied at planting and supplemented before fruit/grain filling or tuber enlargement stages.",
        "environmental_benefits": "Healthy potassium nutrition increases water-use efficiency by crops, reducing water consumption during dry spells.",
        "sustainable_tips": "Recycle crop residues (such as rice or wheat straw, rich in potassium) back into the soil to reduce dependency on imported potash.",
        "type": "Chemical / Mineral",
    },
    "10-26-26": {
        "full_name": "NPK Complex (10-26-26)",
        "npk_ratio": "10-26-26",
        "reason": "The soil shows simultaneous moderate-to-low levels of Phosphorus and Potassium with moderate Nitrogen. Excellent for root/tuber development and flowering.",
        "soil_context": "Recommended for crops like potato, tomato, and sugarcane when both P and K need replenishment.",
        "usage_guidance": "Apply as a basal dose during field preparation or at early tillering / vegetative branching.",
        "environmental_benefits": "Using an integrated complex granule ensures uniform nutrient distribution, avoiding unbalanced single-element toxicity.",
        "sustainable_tips": "Combine with organic mulching to keep root-zone moisture stable and maximize nutrient absorption efficiency.",
        "type": "Complex Mineral",
    },
    "14-35-14": {
        "full_name": "NPK Complex (14-35-14)",
        "npk_ratio": "14-35-14",
        "reason": "High phosphorus requirement with a balanced starter amount of Nitrogen and Potassium. Promotes deep root establishment and robust floral initiation.",
        "soil_context": "Recommended for high phosphorus demanding crops or phosphorus-fixing soils during early growth.",
        "usage_guidance": "Apply at seeding or transplanting in direct contact or bands near root zones.",
        "environmental_benefits": "High phosphorus density in a single granule reduces repeated machine runs, lowering field compaction and fuel emissions.",
        "sustainable_tips": "Maintain adequate soil organic matter to prevent phosphorus from becoming chemically locked up with iron or calcium.",
        "type": "Complex Mineral",
    },
    "20-20-20": {
        "full_name": "Balanced NPK (20-20-20)",
        "npk_ratio": "20-20-20",
        "reason": "The soil shows an overall balanced deficiency where Nitrogen, Phosphorus, and Potassium are all moderately low. Provides equal proportions of all 3 macronutrients.",
        "soil_context": "Recommended when soil tests reveal general nutrient depletion across N, P, and K.",
        "usage_guidance": "Can be applied as a basal dressing or dissolved for foliar spray/fertigation during active vegetative growth.",
        "environmental_benefits": "Balanced nutrition prevents over-supplying one element, which preserves natural microbial equilibrium in the rhizosphere.",
        "sustainable_tips": "Follow the '4R Nutrient Stewardship' concept: Right Source, Right Rate, Right Time, and Right Place.",
        "type": "Balanced Complex",
    },
    "Compost": {
        "full_name": "Well-Decomposed Organic Compost",
        "npk_ratio": "Variable (approx. 1.5 - 0.5 - 1.0 + rich organic carbon)",
        "reason": "Soil nutrient values are in a healthy, balanced range or slightly depleted. Adding compost maintains long-term biological vitality without risking chemical salinity.",
        "soil_context": "Ideal for well-maintained soils, organic farming, or baseline soil conditioning under neutral pH (6.0 - 7.5).",
        "usage_guidance": "Mix thoroughly into the top 15 cm of soil 2–3 weeks before planting or use as side-dressing around growing plants.",
        "environmental_benefits": "Significantly sequesters atmospheric carbon into the soil, improves water-holding capacity, and eliminates chemical runoff.",
        "sustainable_tips": "Produce on-farm vermicompost or aerobic compost from crop stalks, vegetable scraps, and livestock bedding to close nutrient loops.",
        "type": "Organic / Bio-fertilizer",
    },
    "Organic Manure": {
        "full_name": "Farmyard Manure (FYM) / Bio-Manure",
        "npk_ratio": "Organic Multi-Nutrient + Humic Acid + Beneficial Microbes",
        "reason": "Soil pH is out of balance (strongly acidic pH < 5.8 or alkaline pH > 7.8) or overall soil biological health needs rejuvenation. Manure acts as a natural buffer.",
        "soil_context": "Strongly recommended when soil pH is skewed or soil organic carbon is low.",
        "usage_guidance": "Spread evenly and plow into soil 3 to 4 weeks prior to seed sowing so beneficial microbes can assimilate and buffer soil acidity/alkalinity.",
        "environmental_benefits": "Improves soil structure and aeration, restores beneficial earthworm populations, and acts as a natural buffer against soil degradation.",
        "sustainable_tips": "Incorporate green manure crops (e.g., Sesbania, Sunn hemp) once every two years to naturally regenerate nitrogen and organic matter.",
        "type": "Organic Amendment",
    },
}


def evaluate_soil_nutrients(nitrogen: float, phosphorus: float, potassium: float, ph: float) -> dict:
    """
    Evaluates individual soil nutrient levels into qualitative categories
    (Low, Optimal, High) and soil pH (Acidic, Neutral, Alkaline).
    Agronomically standard demonstration benchmarks.
    """
    # Nitrogen benchmarks (kg/ha or ppm)
    if nitrogen < 30:
        n_status = "Low"
        n_color = "red"
        n_desc = "Deficient: Plant foliage may turn pale green or yellow."
    elif nitrogen <= 70:
        n_status = "Optimal"
        n_color = "green"
        n_desc = "Healthy: Supports normal vegetative growth."
    else:
        n_status = "High"
        n_color = "orange"
        n_desc = "Excessive: Risk of lodging, delayed fruiting, and nitrogen leaching."

    # Phosphorus benchmarks
    if phosphorus < 20:
        p_status = "Low"
        p_color = "red"
        p_desc = "Deficient: Slow root development and delayed flowering."
    elif phosphorus <= 50:
        p_status = "Optimal"
        p_color = "green"
        p_desc = "Healthy: Adequate for strong root system and seed formation."
    else:
        p_status = "High"
        p_color = "orange"
        p_desc = "Excessive: Can lock up zinc and iron micronutrients in soil."

    # Potassium benchmarks
    if potassium < 20:
        k_status = "Low"
        k_color = "red"
        k_desc = "Deficient: Weak stalks, leaf margin browning, low drought tolerance."
    elif potassium <= 50:
        k_status = "Optimal"
        k_color = "green"
        k_desc = "Healthy: Optimal water regulation and disease resistance."
    else:
        k_status = "High"
        k_color = "orange"
        k_desc = "Excessive: Can interfere with magnesium and calcium absorption."

    # Soil pH benchmarks
    if ph < 6.0:
        ph_status = "Acidic"
        ph_color = "red"
        ph_desc = "Acidic soil: Reduces availability of P, Ca, and Mg; increases Al toxicity risk."
    elif ph <= 7.5:
        ph_status = "Neutral (Optimal)"
        ph_color = "green"
        ph_desc = "Ideal soil pH: Most nutrients are readily soluble and absorbable."
    else:
        ph_status = "Alkaline"
        ph_color = "red"
        ph_desc = "Alkaline soil: Reduces availability of micronutrients like Iron and Zinc."

    return {
        "Nitrogen": {"value": nitrogen, "status": n_status, "color": n_color, "desc": n_desc},
        "Phosphorus": {"value": phosphorus, "status": p_status, "color": p_color, "desc": p_desc},
        "Potassium": {"value": potassium, "status": k_status, "color": k_color, "desc": k_desc},
        "pH": {"value": ph, "status": ph_status, "color": ph_color, "desc": ph_desc},
    }


def train_model(data_path: str = DATASET_PATH, model_path: str = MODEL_PATH):
    """
    Trains a Decision Tree Classifier on the fertilizer dataset
    and persists the trained model artifact to disk.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)

    # Encode categorical crop feature
    crop_encoder = LabelEncoder()
    df["Crop_Encoded"] = crop_encoder.fit_transform(df["Crop_Type"])

    # Encode categorical target fertilizer
    fertilizer_encoder = LabelEncoder()
    df["Fertilizer_Encoded"] = fertilizer_encoder.fit_transform(df["Fertilizer"])

    # Features and Target
    feature_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Crop_Encoded"]
    X = df[feature_cols]
    y = df["Fertilizer_Encoded"]

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize Decision Tree Classifier (simple, interpretable, beginner-friendly)
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=3,
        random_state=42,
    )

    # Train model
    clf.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Feature importance
    feature_importance = dict(zip(feature_cols, clf.feature_importances_))

    # Package model artifacts
    package = {
        "model": clf,
        "crop_encoder": crop_encoder,
        "fertilizer_encoder": fertilizer_encoder,
        "feature_cols": feature_cols,
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "crops": list(crop_encoder.classes_),
        "fertilizers": list(fertilizer_encoder.classes_),
    }

    # Save to disk
    with open(model_path, "wb") as f:
        pickle.dump(package, f)

    return package


def load_or_train_model(data_path: str = DATASET_PATH, model_path: str = MODEL_PATH):
    """
    Loads an existing trained model or trains a new one if not found.
    """
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                package = pickle.load(f)
            return package
        except Exception:
            # If pickle loading fails, retrain
            return train_model(data_path, model_path)
    else:
        return train_model(data_path, model_path)


def predict_fertilizer(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    crop_type: str,
    model_package: dict = None,
) -> dict:
    """
    Makes a fertilizer recommendation given soil inputs and crop type.
    Returns complete recommendation details including soil status, reasoning,
    sustainable usage guidance, and environmental benefits.
    """
    if model_package is None:
        model_package = load_or_train_model()

    crop_encoder = model_package["crop_encoder"]
    fertilizer_encoder = model_package["fertilizer_encoder"]
    clf = model_package["model"]

    # Input validation
    if crop_type not in crop_encoder.classes_:
        raise ValueError(
            f"Unsupported crop: '{crop_type}'. Supported crops: {list(crop_encoder.classes_)}"
        )
    if not (0.0 <= ph <= 14.0):
        raise ValueError("Soil pH must be between 0.0 and 14.0.")
    if nitrogen < 0 or phosphorus < 0 or potassium < 0:
        raise ValueError("Nutrient values (N, P, K) cannot be negative.")

    # Encode crop
    crop_code = crop_encoder.transform([crop_type])[0]

    # Predict using DataFrame to avoid feature name mismatch warnings
    input_df = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, ph, crop_code]],
        columns=model_package["feature_cols"],
    )
    prediction_idx = clf.predict(input_df)[0]
    fertilizer_name = fertilizer_encoder.inverse_transform([prediction_idx])[0]

    # Predict probabilities
    probs = clf.predict_proba(input_df)[0]
    confidence = float(np.max(probs))

    # Evaluate soil nutrients
    soil_status = evaluate_soil_nutrients(nitrogen, phosphorus, potassium, ph)

    # Retrieve agronomic knowledge
    fert_info = FERTILIZER_KNOWLEDGE_BASE.get(
        fertilizer_name,
        {
            "full_name": fertilizer_name,
            "npk_ratio": "Standard formulation",
            "reason": f"Based on your soil tests, {fertilizer_name} supplies the primary deficient nutrients needed by {crop_type}.",
            "soil_context": "Tailored for the selected soil nutrient levels.",
            "usage_guidance": "Consult a local agricultural extension office for region-specific application timing.",
            "environmental_benefits": "Optimized fertilizer use avoids nutrient pollution and soil degradation.",
            "sustainable_tips": "Incorporate organic matter to build soil health naturally.",
            "type": "Standard Fertilizer",
        },
    )

    return {
        "inputs": {
            "crop": crop_type,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "ph": ph,
        },
        "fertilizer": fertilizer_name,
        "full_name": fert_info["full_name"],
        "npk_ratio": fert_info["npk_ratio"],
        "type": fert_info["type"],
        "confidence": confidence,
        "reason": fert_info["reason"],
        "soil_context": fert_info["soil_context"],
        "usage_guidance": fert_info["usage_guidance"],
        "environmental_benefits": fert_info["environmental_benefits"],
        "sustainable_tips": fert_info["sustainable_tips"],
        "soil_status": soil_status,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Training Sustainable Fertilizer Recommendation Model...")
    print("=" * 60)

    pkg = train_model()
    print(f"Model Training Complete!")
    print(f"Decision Tree Accuracy on Test Set: {pkg['accuracy'] * 100:.2f}%")
    print(f"Feature Importances:")
    for feat, imp in pkg["feature_importance"].items():
        print(f"  - {feat:15s}: {imp:.4f}")

    print("\n--- Testing Sample Prediction ---")
    test_result = predict_fertilizer(
        nitrogen=22, phosphorus=45, potassium=40, ph=6.5, crop_type="Rice", model_package=pkg
    )
    print(f"Crop: {test_result['inputs']['crop']}")
    print(f"Soil N-P-K-pH: N={test_result['inputs']['nitrogen']}, P={test_result['inputs']['phosphorus']}, K={test_result['inputs']['potassium']}, pH={test_result['inputs']['ph']}")
    print(f"Recommended Fertilizer: {test_result['fertilizer']} ({test_result['full_name']})")
    print(f"Reason: {test_result['reason']}")
    print(f"Confidence: {test_result['confidence'] * 100:.1f}%")
    print("=" * 60)
