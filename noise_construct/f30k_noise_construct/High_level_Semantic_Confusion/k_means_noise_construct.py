import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random

# Input file path
file_path = '/path/dataset/k_means_f30k/annotations/scan_split/0_noise_train_caps.txt''/
with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines()]


image_descriptions = []
for i in range(0, len(lines), 5):
    image_texts = lines[i:i+5]
    image_descriptions.append(image_texts)

combined_descriptions = [' '.join(descriptions) for descriptions in image_descriptions]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(combined_descriptions)

n_clusters = 500 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

clustered_images = pd.DataFrame({
    'image_id': range(len(image_descriptions)),
    'descriptions': image_descriptions,
    'cluster': clusters
})


replace_ratio = 1.0  # Replacement ratio
replacements = []  

modified_descriptions = [desc.copy() for desc in image_descriptions]

for cluster in range(n_clusters):
    cluster_images = clustered_images[clustered_images['cluster'] == cluster]
    cluster_indices = cluster_images['image_id'].tolist()
    
    if len(cluster_indices) > 1:  
        for img_idx in cluster_indices:
            for desc_idx in range(5):  
                if random.random() < replace_ratio:
                    other_images = [i for i in cluster_indices if i != img_idx]
                    if other_images:  
                        replacement_img_idx = random.choice(other_images)
                        
                        original_text = image_descriptions[img_idx][desc_idx]
                        replacement_text = image_descriptions[replacement_img_idx][desc_idx]
                        
                        modified_descriptions[img_idx][desc_idx] = replacement_text
                        replacements.append((img_idx, desc_idx, original_text, replacement_text))

# Output file path
output_file = f'/path/dataset/k_means_f30k/annotations/scan_split/{replace_ratio}_noise_train_caps.txt' 
with open(output_file, 'w', encoding='utf-8') as f:
    for image_desc in modified_descriptions:
        for text in image_desc:
            f.write(f"{text}\n")

print(f"Replaced data has been saved to {output_file}")
print(f"Total images processed: {len(image_descriptions)}")
print(f"Total text replacements performed: {len(replacements)}")
