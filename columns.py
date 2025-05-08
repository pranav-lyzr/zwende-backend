import pandas as pd

# OrderExport230425.xlsx
orders = pd.read_excel("data/OrderExport230425.xlsx")
print("OrderExport230425.xlsx columns:")
print(orders.columns.tolist())

# ProductExport230425.xlsx
products = pd.read_excel("data/ProductExport230425.xlsx")
print("\nProductExport230425.xlsx columns:")
print(products.columns.tolist())

# Smart Collections.csv
smart = pd.read_csv("data/Smart Collections.csv")
print("\nSmart Collections.csv columns:")
print(smart.columns.tolist())
print("\nSmart Collections.csv dtypes:")
print(smart.dtypes)
print("\nSmart Collections.csv problematic columns (4, 6, 8, 11, 14):")
print(smart.iloc[:, [4, 6, 8, 11, 14]].head())




# 1. Top Products by Orders in a Category
# bash
# Copy
# Edit
# curl -X GET "http://localhost:8000/search/orders/Art%20%26%20Crafting%20Materials" \
#      -H "Accept: application/json"
# source: orders

# category: Art & Crafting Materials (URL-encoded %20 for spaces, %26 for &)

# Returns: Up to 10 items sold most in that category.

# 2. Top Products in a Smart Collection
# bash
# Copy
# Edit
# curl -X GET "http://localhost:8000/search/collections/Summer%20Favorites" \
#      -H "Accept: application/json"
# source: collections

# category: Summer Favorites (example collection name)

# Returns: Up to 10 items in the “Summer Favorites” smart collection.

# 3. Case-Insensitive Lookup
# bash
# Copy
# Edit
# curl -X GET "http://localhost:8000/search/orders/electronics" \
#      -H "Accept: application/json"
# Even if your DataFrame has "Electronics" or "ELECTRONICS", passing electronics will match.

# 4. 404 When Not Found
# bash
# Copy
# Edit
# curl -i "http://localhost:8000/search/orders/Nonexistent%20Category"
# Response:

# pgsql
# Copy
# Edit
# HTTP/1.1 404 Not Found
# Content-Type: application/json

# {"detail":"No products found for orders - Nonexistent Category"}
# 5. (Optional) Semantic Search Stub
# bash
# Copy
# Edit
# curl -X GET "http://localhost:8000/search/query?q=boho%20baskets&source=orders&top_k=5" \
#      -H "Accept: application/json"
# q: Free-text user query (boho baskets)

# source: orders or collections

# top_k: How many initial items to consider (defaults to 10)

# Response:

# json
# Copy
# Edit
# {"answer":"LLM semantic ranking not configured. Enable OpenAI and uncomment code."}
# Feel free to swap in real category or collection names that exist in your data, and adjust top_k on the /search/query endpoint.