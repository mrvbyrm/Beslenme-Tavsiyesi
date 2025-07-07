from flask import Flask, request, render_template, session
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import os
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Flask uygulaması
app = Flask(__name__)

# CSV dosyasını yükleyin
food_data = pd.read_csv('food.csv', encoding="utf-8", on_bad_lines="skip")
kelime_anlam_data = pd.read_csv('kelime_anlam.csv', encoding="utf-8")

# Sentence-Transformers modelini yükleyin
model = SentenceTransformer('all-MiniLM-L6-v2')

# Yemek açıklamalarını modelle vektöre dönüştürme
food_data['Description'] = food_data['Besin'] + " " + food_data['Tur'] + " " + food_data['Kategori']
food_embeddings = model.encode(food_data['Description'].tolist())

# spaCy NLP modelini yükleyin
nlp = spacy.load("en_core_web_sm")

# İyileştirme için PhraseMatcher kullanacağız
matcher = PhraseMatcher(nlp.vocab)

# Kelime ve anlam eşlemesini kelime_anlam.csv dosyasından alalım
kelime_anlam_dict = {}
for _, row in kelime_anlam_data.iterrows():
    kelime_anlam_dict[row['Yazi'].lower()] = (row['Anlam'], row['Baslik'])

# Yemek türlerini ve öğün tiplerini içeren anahtar kelimeler
meal_types = ["sabaha", "sabah", "öğle", "akşam", "kahvaltı", "öğle yemeği", "akşam yemeği"]
food_types = ["vegan", "vejetaryen", "hiçbiri"]

# PhraseMatcher'a anahtar kelimeleri ekleyelim
meal_patterns = [nlp.make_doc(meal) for meal in meal_types]
food_patterns = [nlp.make_doc(food) for food in food_types]

matcher.add("MEAL_TYPE", None, *meal_patterns)
matcher.add("FOOD_TYPE", None, *food_patterns)

def extract_preferences_simple(user_input):
    """Kullanıcının girdiği metinden öğün tipini ve yemek türünü çıkarır (basit kontrol)."""
    meal_type = None
    food_type = None

    # Girdi metnini küçük harfe çevir
    user_input_lower = user_input.lower()

    # Öğün türlerini kontrol et
    for meal in meal_types:
        if meal in user_input_lower:
            meal_type = meal
            break

    # Yemek türlerini kontrol et
    for food in food_types:
        if food in user_input_lower:
            food_type = food
            break

    return food_type, meal_type


def nlp_based_recommendation_simple(user_input):
    """Kullanıcının tercihine ve öğün tipine göre yemek önerir (Basit arama)."""
    # Kullanıcının tercih ve öğün tipini çıkar
    food_type, meal_type = extract_preferences_simple(user_input)

    # Eğer yemek türü veya öğün tipi çıkartılamazsa, hata mesajı döndür
    if not food_type or not meal_type:
        return "Lütfen geçerli bir yemek türü (vegan, vejetaryen, vb.) ve öğün tipi (Sabah, Öğle, Akşam) belirtin."

    # Veriyi filtrele: Kullanıcının tercihine ve öğün tipine göre
    filtered_foods = food_data[(
        food_data['Tur'].str.lower() == food_type) & (food_data['Ogun'].str.lower() == meal_type.lower())]

    if filtered_foods.empty:
        return f"Hiçbir {food_type} yemek bulunamadı."

    # Kullanıcı girişi ile her yemek açıklaması arasındaki benzerliği hesapla
    user_embedding = model.encode([user_input])[0]
    cosine_similarities = cosine_similarity([user_embedding], food_embeddings)[0]

    # En yüksek benzerliğe sahip yemekleri seç (3 yemek al)
    top_indices = cosine_similarities.argsort()[-3:][::-1]  # En yüksek 3 yemek

    # Önerilen yemekleri listele, "Ogun" sütunu hariç
    recommended_foods = food_data.iloc[top_indices][
        ['Besin', 'Tur', 'Kategori', 'Kalori', 'Protein', 'Karbonhidrat', 'Yağ', 'Kolesterol']]

    # Sonuçları HTML tablosu olarak döndür
    return recommended_foods.to_html(index=False)



def get_word_mapping(user_input):
    """Kullanıcı cümlesindeki kelimeleri kelime_anlam.csv'den çekip anlamlarını döndürür."""
    words = user_input.lower().split()
    word_mapping = []
    for word in words:
        if word in kelime_anlam_dict:
            word_mapping.append(kelime_anlam_dict[word])
    return word_mapping


def calculate_calories_and_macros(age, gender, weight, height, activity_level):
    """Günlük kalori ihtiyacı ve makro besinleri hesaplar."""
    # Bazal metabolizma hızı (BMR) hesaplama
    if gender == "erkek":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Aktivite seviyesine göre çarpan
    activity_multipliers = {
        "düşük": 1.2,
        "orta": 1.55,
        "yüksek": 1.9
    }
    calorie_needs = bmr * activity_multipliers[activity_level]

    # Makro besin dağılımı (protein: %20, karbonhidrat: %50, yağ: %30)
    macros = {
        "protein": round((calorie_needs * 0.2) / 4),
        "carbs": round((calorie_needs * 0.5) / 4),
        "fats": round((calorie_needs * 0.3) / 9)
    }

    return round(calorie_needs), macros

def calculate_bmi(weight, height):
    """Vücut Kitle İndeksini (VKİ) hesaplar."""
    height_in_meters = height / 100
    bmi = weight / (height_in_meters ** 2)

    # VKI kategorisi belirleme
    if bmi < 18.5:
        category = "Zayıf"
    elif 18.5 <= bmi < 24.9:
        category = "İdeal kiloda"
    elif 25 <= bmi < 29.9:
        category = "Fazla kilolu"
    else:
        category = "Obez"

    # İdeal kilo aralığı hesaplama
    ideal_weight_min = round(18.5 * (height_in_meters ** 2), 1)
    ideal_weight_max = round(24.9 * (height_in_meters ** 2), 1)

    return round(bmi, 1), category, ideal_weight_min, ideal_weight_max

def generate_diet_plan(calorie_needs):
    """Günlük kalori ihtiyacına göre sabah, öğle ve akşam için diyet planı oluşturur."""
    # Her öğün için uygun yemekleri seç
    breakfast_foods = food_data[food_data['Ogun'].str.lower() == 'sabah']
    lunch_foods = food_data[food_data['Ogun'].str.lower() == 'öğle']
    dinner_foods = food_data[food_data['Ogun'].str.lower() == 'akşam']

    # Her öğün için farklı kategorilerden yemek seçimi
    breakfast = select_food_for_meal_with_category_control(breakfast_foods, calorie_needs * 0.33, [])
    lunch = select_food_for_meal_with_category_control(lunch_foods, calorie_needs * 0.33, [breakfast["Kategori"] if breakfast is not None else None])
    dinner = select_food_for_meal_with_category_control(dinner_foods, calorie_needs * 0.34, [
        breakfast["Kategori"] if breakfast is not None else None,
        lunch["Kategori"] if lunch is not None else None
    ])

    # Sadece gerekli sütunları dahil et (Besin, Kalori, Protein, Karbonhidrat, Yağ, Kolesterol)
    cols_to_display = ['Besin', 'Kalori', 'Protein', 'Karbonhidrat', 'Yağ', 'Kolesterol']

    # HTML tablosu olarak döndür
    return {
        "sabah": breakfast[cols_to_display].to_html(index=False) if breakfast is not None else "Uygun sabah yemeği bulunamadı.",
        "öğle": lunch[cols_to_display].to_html(index=False) if lunch is not None else "Uygun öğle yemeği bulunamadı.",
        "akşam": dinner[cols_to_display].to_html(index=False) if dinner is not None else "Uygun akşam yemeği bulunamadı."
    }

def select_food_for_meal_with_category_control(food_data, calorie_limit, used_categories):
    """Belirtilen kalori limitine uyan ve kullanılmayan kategorilerden rastgele bir yemek seçer."""
    selected_foods = []  # Birden fazla yemek eklemek için liste
    total_calories = 0

    # Kullanılmış kategorileri hariç tut
    available_foods = food_data[~food_data['Kategori'].isin(used_categories)]

    if available_foods.empty:
        return None  # Eğer uygun yemek kalmadıysa, None döndür

    # Kalori limiti ile uyumlu yemekler seçmek
    while total_calories < calorie_limit and not available_foods.empty:
        random_food = available_foods.sample(n=1).iloc[0]
        if total_calories + random_food['Kalori'] <= calorie_limit + 100:  # Biraz esneklik
            selected_foods.append(random_food)
            total_calories += random_food['Kalori']
        available_foods = available_foods.drop(random_food.name)  # Yemeği listeden çıkar

    if not selected_foods:
        return None  # Eğer uygun bir yemek bulunamazsa, None döndür

    # Seçilen yemekleri DataFrame olarak döndür
    selected_foods_df = pd.DataFrame(selected_foods)
    return selected_foods_df



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health_check', methods=['POST'])
def health_check():
    if 'age' in request.form:  # İlk POST isteği (Formdan gelen veriler)
        session['age'] = int(request.form['age'])
        session['gender'] = request.form['gender']
        session['weight'] = float(request.form['weight'])
        session['height'] = float(request.form['height'])
        session['activity_level'] = request.form['activity_level']

    # Oturumdan parametreleri al
    age = session.get('age')
    gender = session.get('gender')
    weight = session.get('weight')
    height = session.get('height')
    activity_level = session.get('activity_level')

    if not all([age, gender, weight, height, activity_level]):
        return "Eksik oturum bilgisi. Lütfen formu yeniden doldurun.", 400

    # Kalori ve makro besin hesaplama
    calories, macros = calculate_calories_and_macros(age, gender, weight, height, activity_level)

    # VKI hesaplama
    bmi, bmi_category, ideal_weight_min, ideal_weight_max = calculate_bmi(weight, height)

    # Diyet planı oluşturma
    diet_plan = generate_diet_plan(calories)

    return render_template(
        'index.html',
        calories=calories,
        macros=macros,
        bmi=bmi,
        bmi_category=bmi_category,
        ideal_weight_range=f"{ideal_weight_min}-{ideal_weight_max} kg",
        breakfast=diet_plan["sabah"],
        lunch=diet_plan["\u00f6\u011fle"],
        dinner=diet_plan["ak\u015fam"],
        refresh=True
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    recommendation = nlp_based_recommendation_simple(user_input)

    # Kelime anlamlarını almak için get_word_mapping fonksiyonunu çağırıyoruz
    word_mapping = get_word_mapping(user_input)

    # Sonuçları kullanıcıya göster
    return render_template('index.html', recommendation=recommendation, word_mapping=word_mapping)

if __name__ == "__main__":
    app.secret_key = 'bu-coook-gizli-bir-anahtar'
    app.run(debug=True)