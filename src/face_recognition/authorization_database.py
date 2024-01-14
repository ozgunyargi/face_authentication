import os

import torch
from torch.nn.functional import cosine_similarity

class AuthorizationDatabase:
    def __init__(self, room_path = 'rooms'):
        self.room_path = room_path

    def create_room(self, room_name):
        room_path = os.path.join(self.room_path, room_name)
        if os.path.exists(room_path):
            print(f'Room {room_name} already exists')
        else:
            os.makedirs(room_path)

    def add_user(self, room_name, user_id, embedding):
        room_path = os.path.join(self.room_path, room_name)
        if not os.path.exists(room_path):
            print(f'Room {room_name} not exists. Creating...')
            os.makedirs(room_path)

        user_embedding_path = os.path.join(room_path, f'{user_id}_emb.pt')
        if os.path.exists(user_embedding_path):
            print(f"User '{user_id}' is already authorized in room '{room_name}'. Please remove it first to add again.")
        else:
            torch.save(embedding, user_embedding_path)
            print(f"User '{user_id}' is added to authorized users for room '{room_name}'.")

    def remove_user(self, room_name, user_id):
        room_path = os.path.join(self.room_path, room_name)
        user_embedding_path = os.path.join(room_path, f'{user_id}_emb.pt')
        if os.path.exists(user_embedding_path):
            os.remove(user_embedding_path)
            print(f"User '{user_id}' has been removed from room '{room_name}'.")
        else:
            print("Failed! User '{user_id}' is not authorized in room '{room_name}'")

    def authorize_user(self, room_name, new_embedding, threshold=0.7):
        room_path = os.path.join(self.room_path, room_name)
        if not os.path.exists(room_path):
            print(f"Room '{room_name}' does not exist.")
            return False

        user_ids = []
        room_embeddings = []
        for filename in os.listdir(room_path):
            if filename.endswith('.pt'):
                user_id, _ = os.path.splitext(filename)
                saved_embedding = torch.load(os.path.join(room_path, filename))
                similarity = cosine_similarity(new_embedding, saved_embedding).item()
                room_embeddings.append(similarity)
                user_ids.append(user_id.split('_')[0])

        if room_embeddings:
            max_similarity_idx = room_embeddings.index(max(room_embeddings))
            max_similarity = max(room_embeddings)
            if max_similarity > threshold:
                authorized_user = user_ids[max_similarity_idx]
                return True, authorized_user, max_similarity
            return False, None, max_similarity
        return False, None, None