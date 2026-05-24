"""
Central configuration for plant disease classes and keywords.
"""

PLANT_DISEASE_CLASSES = {
    "Rice_Healthy": {
        "plant": "Rice",
        "disease": "Healthy",
        "keywords": [
            "healthy rice leaf",
            "rice plant healthy leaf",
            "green rice leaves",
            "healthy paddy leaf",
            "lá lúa khỏe mạnh",
            "cây lúa khỏe mạnh"
        ]
    },
    "Rice_LeafBlast": {
        "plant": "Rice",
        "disease": "Leaf Blast",
        "keywords": [
            "rice leaf blast disease",
            "rice blast leaf symptoms",
            "Magnaporthe oryzae rice leaf",
            "rice blast lesions",
            "rice leaf blast close up",
            "bệnh đạo ôn lá lúa",
            "đạo ôn trên lá lúa"
        ]
    },
    "Rice_BrownSpot": {
        "plant": "Rice",
        "disease": "Brown Spot",
        "keywords": [
            "rice brown spot disease leaf",
            "brown spot rice leaf symptoms",
            "Bipolaris oryzae rice leaf",
            "rice leaf brown lesions",
            "bệnh đốm nâu lá lúa",
            "đốm nâu trên lá lúa"
        ]
    },
    "Tomato_Healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "keywords": [
            "healthy tomato leaf",
            "tomato plant healthy leaves",
            "green tomato leaves",
            "healthy tomato plant leaf",
            "lá cà chua khỏe mạnh"
        ]
    },
    "Tomato_EarlyBlight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "keywords": [
            "tomato early blight leaf",
            "Alternaria solani tomato leaf",
            "early blight tomato symptoms",
            "tomato leaf concentric rings disease",
            "tomato early blight close up",
            "bệnh cháy lá sớm cà chua"
        ]
    },
    "Tomato_LateBlight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "keywords": [
            "tomato late blight leaf",
            "Phytophthora infestans tomato leaf",
            "late blight tomato symptoms",
            "tomato leaf water soaked lesions",
            "tomato late blight disease close up",
            "bệnh mốc sương cà chua"
        ]
    },
    "Tomato_LeafMold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "keywords": [
            "tomato leaf mold disease",
            "Passalora fulva tomato leaf",
            "tomato leaf mold symptoms",
            "yellow spots tomato leaf mold",
            "mold on tomato leaves",
            "bệnh mốc lá cà chua"
        ]
    },
    "Tomato_SeptoriaLeafSpot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "keywords": [
            "tomato septoria leaf spot",
            "Septoria lycopersici tomato leaf",
            "tomato leaf small circular spots",
            "septoria tomato symptoms",
            "tomato septoria leaf disease",
            "đốm lá septoria cà chua"
        ]
    },
    "Potato_Healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "keywords": [
            "healthy potato leaf",
            "potato plant healthy leaves",
            "green potato leaves",
            "healthy potato foliage",
            "lá khoai tây khỏe mạnh"
        ]
    },
    "Potato_EarlyBlight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "keywords": [
            "potato early blight leaf",
            "Alternaria solani potato leaf",
            "early blight potato symptoms",
            "potato leaf concentric rings",
            "potato early blight disease",
            "bệnh cháy lá sớm khoai tây"
        ]
    },
    "Potato_LateBlight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "keywords": [
            "potato late blight leaf",
            "Phytophthora infestans potato leaf",
            "late blight potato symptoms",
            "potato leaf water soaked lesions",
            "potato late blight disease",
            "bệnh mốc sương khoai tây"
        ]
    },
    "Corn_Healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "keywords": [
            "healthy corn leaf",
            "maize healthy leaf",
            "green corn leaves",
            "healthy maize plant",
            "lá ngô khỏe mạnh",
            "lá bắp khỏe mạnh"
        ]
    },
    "Corn_CommonRust": {
        "plant": "Corn",
        "disease": "Common Rust",
        "keywords": [
            "corn common rust leaf",
            "maize common rust disease",
            "Puccinia sorghi corn leaf",
            "corn leaf rust pustules",
            "common rust on corn leaves",
            "bệnh gỉ sắt lá ngô"
        ]
    },
    "Corn_NorthernLeafBlight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "keywords": [
            "corn northern leaf blight",
            "maize northern leaf blight",
            "Exserohilum turcicum corn leaf",
            "corn leaf long gray lesions",
            "northern corn leaf blight symptoms",
            "bệnh cháy lá ngô phía bắc"
        ]
    },
    "Apple_Healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "keywords": [
            "healthy apple leaf",
            "apple tree healthy leaves",
            "green apple leaves",
            "healthy apple foliage",
            "lá táo khỏe mạnh"
        ]
    },
    "Apple_Scab": {
        "plant": "Apple",
        "disease": "Scab",
        "keywords": [
            "apple scab leaf",
            "Venturia inaequalis apple leaf",
            "apple scab disease symptoms",
            "apple leaf olive spots",
            "apple scab close up",
            "bệnh ghẻ lá táo"
        ]
    }
}

CLASS_NAMES = sorted(list(PLANT_DISEASE_CLASSES.keys()))
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}