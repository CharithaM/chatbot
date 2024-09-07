import random
import json
import uuid
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

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

# Drop all existing tables and recreate them
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Create a sessionmaker bound to the engine
Session = sessionmaker(bind=engine)
session = Session()

# Function to add products to the database
def add_products(products):
    for product in products:
        new_product = Product(name=product)
        session.add(new_product)
    session.commit()

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

conversation_state = {}

def get_response(msg, user_id):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

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
                    conversation_state[user_id] = {'state': 'ordering'}
                    print(f"Sure! Please choose a product from the list below:\n{', '.join(product_names)}")
                    product_name = input("Product: ").strip().lower()
                    product = session.query(Product).filter(Product.name.ilike(f"%{product_name}%")).first()
                    if product:
                        order_number = str(uuid.uuid4())
                        new_order = Order(order_number=order_number, product_id=product.id)
                        session.add(new_order)
                        session.commit()
                        conversation_state[user_id]['state'] = 'idle'
                        return f"Your order has been placed! Order number: {order_number}"
                    else:
                        return "Sorry, we don't have that product. Please choose from the list."
                
                else:
                    return random.choice(intent['responses'])
    
    return "I do not understand..."

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        user_id = str(uuid.uuid4())  # Simulate unique user ID for conversation state
        resp = get_response(sentence, user_id)
        print("Bot:", resp)

# Close the session when done
session.close()
