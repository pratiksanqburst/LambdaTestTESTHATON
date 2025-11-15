import requests

BASE_URL = "https://reqres.in/api"

def list_users(page=1):
    url = f"{BASE_URL}/users"
    params = {"page": page}
    resp = requests.get(url, params=params)
    print("List users:", resp.status_code, resp.json())
    return resp

def get_user(user_id):
    url = f"{BASE_URL}/users/{user_id}"
    resp = requests.get(url)
    print(f"Get user {user_id}:", resp.status_code, resp.json())
    return resp

def create_user(name, job):
    url = f"{BASE_URL}/users"
    data = {"name": name, "job": job}
    resp = requests.post(url, json=data)
    print("Create user:", resp.status_code, resp.json())
    return resp

def login(email, password):
    url = f"{BASE_URL}/login"
    data = {"email": email, "password": password}
    resp = requests.post(url, json=data)
    print("Login:", resp.status_code, resp.json())
    return resp

if __name__ == "__main__":
    # example calls
    list_users(page=1)
    get_user(2)
    get_user(23)       # non-existent user
    create_user("Jane Doe", "Tester")
    login("eve.holt@reqres.in", "cityslicka")
    login("wrong@reqres.in", "wrongpass")