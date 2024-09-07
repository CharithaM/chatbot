from flask import Flask, request, jsonify,render_template
import torch
import json
import uuid
from model import NeuralNet
import random
from nltk_utils import bag_of_words, tokenize
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

app = Flask(__name__)

food_check = 0
# Load intents from JSON
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Create SQLite database engine
engine = create_engine('sqlite:///shop.db', echo=False)

# Define a base class for declarative class definitions
Base = declarative_base()

# Define Product model
class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    name = Column(String)

# Define Order model
class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    order_number = Column(String, unique=True)
    product_id = Column(Integer, ForeignKey('products.id'))

    # Relationship to Product
    product = relationship('Product')

# Base.metadata.drop_all(engine)
# Base.metadata.create_all(engine)

def create_tables_if_not_exists():
    # Create tables if they do not exist
    Base.metadata.create_all(engine)

# Call this function once during application startup
create_tables_if_not_exists()

Session = sessionmaker(bind=engine)

def get_session():
    return Session()

# Function to add products to the database
def add_products(products):
    session = get_session()  # Create a new session
    try:
        for product in products:
            new_product = Product(name=product)
            session.add(new_product)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    finally:
        session.close()  # Ensure the session is closed

# Example of adding products
products_to_add = ['Coffee', 'Tea', 'Milk', 'Sugar', 'Cookies']
add_products(products_to_add)


# Load pretrained model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

food_check = {'value': 0}

def is_food_item(product_name):
    session = get_session()
    try:
        product_name = product_name.strip().lower()
        product = session.query(Product).filter(Product.name.ilike(f"%{product_name}%")).first()
        return product is not None
    finally:
        session.close()


def get_response(msg, user_id):
    if food_check['value'] == 1 and is_food_item(msg):
        session = get_session()
        try:
            product = session.query(Product).filter(Product.name.ilike(f"%{msg.strip().lower()}%")).first()
            if product:
                # Use the order ID as the row number
                new_order_number = session.query(Order).count() + 1
                new_order = Order(order_number=str(new_order_number), product_id=product.id)
                session.add(new_order)
                session.commit()
                food_check['value'] = 0
                return f"Your order has been placed! Order number: {new_order_number}"
            else:
                return "The item you requested is not available."
        finally:
            session.close()
        
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    session = get_session()
    try:
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if intent['tag'] == 'items':
                        products = session.query(Product.name).distinct().all()
                        product_names = [product.name for product in products]
                        return f"We have {', '.join(product_names)}"
                    
                    elif intent['tag'] == 'order':
                        products = session.query(Product.name).distinct().all()
                        product_names = [product.name for product in products]
                        food_check['value'] = 1
                        return f"Sure! Please choose a product from the list below:\n{', '.join(product_names)}"
                    
                    else:
                        return random.choice(intent['responses'])
    finally:
        session.close()

    return "I do not understand..."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response_route():
    user_message = request.json.get('message', '')
    user_id = str(uuid.uuid4())  # Simulate unique user ID for conversation state
    response = get_response(user_message, user_id)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

# Close the session when done
@app.teardown_appcontext
def shutdown_session(exception=None):
    session.remove()
