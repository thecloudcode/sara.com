import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time
from textblob import TextBlob  # For sentiment analysis

nlp = spacy.load("en_core_web_sm")

ecommerce_items = {
    'electronics': ['laptop', 'smartphone', 'tablet', 'camera', 'headphones', 'smartwatch', 'television', 'gaming console', 'printer', 'router', 'bluetooth speaker', 'monitor', 'mouse', 'keyboard', 'drone'],
    'clothing': ['shirt', 'pants', 'jacket', 'dress', 'shoes', 'hat', 'scarf', 'gloves', 'sweater', 'jeans', 't-shirt', 'coat', 'socks', 'skirt', 'blouse'],
    'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'wardrobe', 'bookshelf', 'coffee table', 'dining table', 'nightstand', 'armchair', 'barstool', 'dresser', 'tv stand', 'shoe rack'],
    'appliances': ['refrigerator', 'washing machine', 'microwave', 'blender', 'air conditioner', 'vacuum cleaner', 'oven', 'toaster', 'dishwasher', 'kettle', 'air purifier', 'water heater', 'iron', 'hair dryer', 'coffee maker'],
    'beauty': ['lipstick', 'foundation', 'mascara', 'eyeliner', 'moisturizer', 'shampoo', 'conditioner', 'hair serum', 'nail polish', 'sunscreen', 'perfume', 'face mask', 'makeup remover', 'body lotion', 'blush'],
    'books': ['novel', 'textbook', 'comic book', 'magazine', 'cookbook', 'biography', 'poetry', 'self-help book', 'children’s book', 'mystery novel', 'science fiction book', 'romance novel', 'horror novel', 'history book', 'non-fiction'],
    'sports equipment': ['tennis racket', 'football', 'basketball', 'yoga mat', 'dumbbell', 'treadmill', 'bicycle', 'helmet', 'golf club', 'soccer ball', 'cricket bat', 'hiking boots', 'skateboard', 'boxing gloves', 'jump rope'],
    'toys': ['lego', 'action figure', 'doll', 'board game', 'puzzle', 'rc car', 'teddy bear', 'play-doh', 'kitchen set', 'water gun', 'train set', 'robot toy', 'bicycle', 'stuffed animal', 'building blocks'],
    'groceries': ['bread', 'milk', 'eggs', 'flour', 'rice', 'pasta', 'sugar', 'tea', 'coffee', 'fruits', 'vegetables', 'meat', 'fish', 'spices', 'cereal'],
    'jewelry': ['necklace', 'bracelet', 'ring', 'earrings', 'watch', 'pendant', 'brooch', 'anklet', 'cufflinks', 'bangle', 'charm', 'tiara', 'hairpin', 'engagement ring', 'wedding band'],
    'automotive': ['car battery', 'tires', 'engine oil', 'brake pads', 'headlights', 'windshield wipers', 'spark plugs', 'car mats', 'car cover', 'air filter', 'seat covers', 'oil filter', 'radiator', 'steering wheel cover', 'bumper'],
    'office supplies': ['notebook', 'pen', 'pencil', 'stapler', 'scissors', 'paper clips', 'printer paper', 'envelope', 'glue', 'sticky notes', 'whiteboard', 'binder', 'marker', 'file folder', 'calendar'],
    'healthcare': ['thermometer', 'blood pressure monitor', 'glucose meter', 'face mask', 'hand sanitizer', 'bandages', 'first aid kit', 'inhaler', 'eye drops', 'heating pad', 'pain relief gel', 'cough syrup', 'antiseptic cream', 'vitamins', 'disposable gloves'],
    'baby products': ['diapers', 'baby wipes', 'stroller', 'baby bottle', 'pacifier', 'baby monitor', 'high chair', 'crib', 'baby formula', 'changing table', 'baby clothes', 'baby carrier', 'diaper bag', 'baby shampoo', 'teething toy'],
    'pet supplies': ['dog food', 'cat food', 'pet bed', 'pet toy', 'dog leash', 'cat litter', 'fish tank', 'pet grooming kit', 'bird cage', 'dog collar', 'aquarium filter', 'pet carrier', 'dog house', 'pet water fountain', 'scratch post'],
}

ecommerce_themes = {
    'promotion': ['discount', 'sale', 'offer', 'promo', 'voucher', 'coupon', 'black friday', 'cyber monday', 'clearance', 'flash sale', 'deal of the day', 'bundle offer', 'limited time offer', 'buy one get one', 'seasonal sale'],
    'shipping': ['delivery', 'shipping', 'courier', 'logistics', 'express delivery', 'same-day delivery', 'tracking number', 'standard shipping', 'free shipping', 'international shipping', 'shipping cost', 'shipping delay', 'shipping status', 'return policy', 'in transit'],
    'pricing': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'premium', 'budget-friendly', 'price match', 'overpriced', 'bargain', 'price drop', 'wholesale', 'retail price', 'price increase', 'market price'],
    'review': ['good', 'bad', 'recommend', 'terrible', 'awesome', '5-star', 'customer feedback', 'user review', 'testimonial', 'product rating', 'satisfaction', 'reviewer', 'quality', 'experience', 'negative feedback'],
    'availability': ['in stock', 'out of stock', 'limited stock', 'pre-order', 'restock', 'sold out', 'inventory', 'available now', 'back in stock', 'stock availability', 'low stock', 'backordered', 'shipping soon', 'limited edition', 'new arrival'],
    'payment': ['credit card', 'debit card', 'paypal', 'bank transfer', 'cash on delivery', 'installment', 'payment method', 'refund', 'invoice', 'checkout', 'payment gateway', 'transaction', 'credit terms', 'payment confirmation', 'secure payment'],
    'returns': ['return policy', 'refund', 'exchange', 'replacement', 'return window', 'warranty', 'defective product', 'return label', 'return process', 'money-back guarantee', 'repair service', 'return shipping', 'reimbursement', 'return request', 'refund status'],
    'customer service': ['support', 'help', 'contact', 'live chat', 'email support', 'call center', 'customer care', 'service agent', 'complaint', 'inquiry', 'response time', 'service desk', 'customer satisfaction', 'technical support', 'issue resolution'],
    'sustainability': ['eco-friendly', 'sustainable', 'recyclable', 'organic', 'biodegradable', 'carbon footprint', 'green packaging', 'environmental impact', 'energy-efficient', 'sustainable sourcing', 'ethical production', 'zero waste', 'fair trade', 'renewable resources', 'plastic-free'],
    'warranty': ['extended warranty', 'guarantee', 'manufacturer warranty', 'coverage', 'warranty claim', 'repair', 'maintenance', 'warranty period', 'lifetime warranty', 'warranty terms', 'damage protection', 'replacement warranty', 'breakage', 'product protection', 'repair service'],
    'brand': ['brand name', 'reputable', 'new brand', 'designer', 'luxury', 'brand loyalty', 'top-rated brand', 'household name', 'emerging brand', 'brand reputation', 'branded product', 'flagship store', 'brand ambassador', 'exclusive brand', 'designer label'],
}

def get_tweet():
    response = requests.get("http://127.0.0.1:8000/tweet")
    return response.json()["text"]

def classify_tweet(tweet):
    for category, items in ecommerce_items.items():
        for item in items:
            if item.lower() in tweet.lower():
                return category
    return "unclassified"

def detect_theme(tweet):
    detected_themes = []
    for theme, keywords in ecommerce_themes.items():
        for keyword in keywords:
            if keyword.lower() in tweet.lower():
                detected_themes.append(theme)
    return detected_themes if detected_themes else ["no theme"]

def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        return "positive"
    elif sentiment_polarity < 0:
        return "negative"
    else:
        return "neutral"

def process_tweet():
    tweet = get_tweet()
    classification = classify_tweet(tweet)
    themes = detect_theme(tweet)
    sentiment = analyze_sentiment(tweet)

    print(f"Tweet: {tweet}")
    print(f"Classified as: {classification}")
    print(f"Themes detected: {themes}")
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    while True:
        time.sleep(1)
        process_tweet()
