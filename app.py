from flask import Flask, render_template, request, send_file
import pandas as pd
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
import time
import csv

app = Flask(__name__)

# ฟังก์ชันสำหรับแปลข้อความ
def translate_text(text, src='th', dest='en'):
    translator = GoogleTranslator(source=src, target=dest)
    return translator.translate(text)


def predict(text, model_path, token_path):
    # แปลข้อความเป็นภาษาอังกฤษ
    translated_text = translate_text(text)
    # โหลดโมเดลและ tokenizer
    model = load_model(model_path)
    with open(token_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # เตรียมข้อมูลสำหรับทำนาย
    sequences = tokenizer.texts_to_sequences([translated_text])
    padded_seq = pad_sequences(sequences, maxlen=50) # อาจต้องปรับ maxlen ให้ตรงกับที่ใช้ในการฝึกโมเดล
    
    # ทำนาย
    predictions = model.predict([padded_seq, padded_seq])
    
    # แสดงผลลัพธ์การทำนาย
    emotions = {0: 'anger', 1: 'fear', 2: 'happiness', 3:'hate', 4:'joy', 5:'sadness'}
    label = list(emotions.values())
    probs = list(predictions[0])
    labels = label
    plt.figure(figsize=(8, 6))
    bars = plt.barh(labels, probs)
    ax = plt.gca()
    ax.bar_label(bars, fmt='%.2f')
    plt.tight_layout()
    # สร้างชื่อไฟล์รูปใหม่โดยใช้เวลาปัจจุบันเป็นส่วนต่อประกอบ
    timestamp = str(int(time.time()))
    prediction_filename = f'static/prediction_{timestamp}.png'

    plt.savefig(prediction_filename)  # บันทึกรูปภาพเป็นไฟล์ใหม่
    plt.close()
    return labels, probs, prediction_filename


@app.context_processor
def utility_processor():
    return dict(zip=zip)

@app.route('/')
def index():
    return render_template('page1.html')

def save_text_to_csv(text):
    # แปลข้อความเป็นภาษาอังกฤษ
    english_text = translate_text(text)
    
    with open('user_input.csv', 'a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([english_text])

def translate_text(text, src='auto', dest='en'):
    translator = GoogleTranslator(source=src, target=dest)
    return translator.translate(text)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        text = request.form['text']
        save_text_to_csv(text)  # เรียกใช้ฟังก์ชันเพื่อบันทึกข้อมูล
        labels, probs, plot_filename = predict(text, r'C:\Users\asus\Desktop\NLP\the_best_model\nlp_balanced_ver.h5', r'C:\Users\asus\Desktop\NLP\the_best_model\tokenizer_balanced_ver.pkl')
        return render_template('result.html', text=text, labels=labels, probs=probs, plot_filename=plot_filename)
    
@app.route('/page2')
def page2():
    
    df = pd.read_excel("movie_comments.xlsx")
    
    return render_template('page2.html', df=df)

@app.route('/download')
def download_file():
    return send_file('movie_comments.xlsx', as_attachment=True)


@app.route('/scrape', methods=['GET'])
def scrape_and_save():
    url = request.args.get('url')  # รับ URL ที่ส่งมาจากแบบฟอร์ม
    if url:
        header1 = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}
        cookies = dict(cookie='value')

        req = requests.get(url, headers=header1, cookies=cookies).text

        soup = BeautifulSoup(req, 'lxml')

        # ค้นหาและดึง HTML element ที่มี attribute reviewquote=
        review_text_elements = soup.find_all('p', class_='review-text')

        # สร้างรายการเพื่อเก็บข้อมูลที่ดึงได้
        comments = []

        # วนลูปเพื่อดึงค่าของ attribute reviewquote แต่ละตัว
        for element in review_text_elements:
            # ดึงข้อความภายในแต่ละ <p> element และเพิ่มลงในรายการ comments
            comment = element.get_text(strip=True)
            comments.append(comment)

        # สร้าง DataFrame จากข้อมูล comments
        df = pd.DataFrame({"comment": comments})

        # บันทึก DataFrame เป็นไฟล์ Excel
        df.to_excel("movie_comments.xlsx", index=False)
        
        # ส่งตัวแปร df ไปยังเทมเพลต page2.html
        return render_template('page2.html', df=df)
    else:
        return "กรุณาใส่ URL ก่อนครับ/ค่ะ"



@app.route('/interactive_prediction', methods=['POST'])
def interactive_prediction():
    choice = int(request.form['choice'])
    
    # อ่านข้อมูลจากไฟล์ Excel
    df = pd.read_excel("movie_comments.xlsx")

    comment = df.loc[choice - 1, 'comment']
    # กำหนดที่อยู่ของโมเดลและโทเคนไนเซอร์
    model_path = r"C:\Users\asus\Desktop\NLP\the_best_model\nlp_balanced_ver.h5"
    tokenizer_path = r"C:\Users\asus\Desktop\NLP\the_best_model\tokenizer_balanced_ver.pkl"
    label, probs, plot_filename2 = predict(comment, model_path, tokenizer_path)


    # สร้างชื่อไฟล์ใหม่ด้วย timestamp เพื่อไม่ให้ซ้ำกัน
    timestamp = str(int(time.time()))
    plot_filename2 = f'static/prediction_plot_{timestamp}.png'


    # สร้างกราฟ
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label, probs, color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Emotion Probability Distribution')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    max_prob_index = probs.index(max(probs))
    max_prob = max(probs)

    for i, prob in enumerate(probs):
        if i != max_prob_index:
            plt.text(i, prob, f'{prob*100:.2f}%', ha='center', va='bottom')

    plt.annotate(f'Max: {max_prob*100:.2f}%', xy=(max_prob_index, max_prob), xytext=(max_prob_index, max_prob + 0.05),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)

    plt.tight_layout()

    # บันทึกกราฟเป็นไฟล์รูปและเก็บที่อยู่ไว้
    plt.savefig(plot_filename2)
    plt.close()

    # ส่งข้อมูลที่จะใช้ในการแสดงผลไปยัง HTML template
    return render_template('page2.html', comment=comment, label=label, probs=probs, plot_filename2=plot_filename2, df=df)

@app.route('/page3', methods=['GET', 'POST'])
def page3():
    if request.method == 'GET':
        return render_template('page3.html')
    elif request.method == 'POST':
        url = request.form['url']
        threshold = float(request.form['threshold'])
        criticsscores, movienames = get_movies_with_critic_scores_above_threshold(url, threshold)
        
        # เรียงลำดับข้อมูลตามคะแนน critic scores
        sorted_data = sorted(zip(criticsscores, movienames), reverse=True)
        criticsscores, movienames = zip(*sorted_data)
        
        return render_template('page3.html', criticsscores=criticsscores, movienames=movienames, threshold=threshold)

def get_movies_with_critic_scores_above_threshold(url, threshold):
    # ส่งคำร้องขอ GET เพื่อเข้าถึงเนื้อหาของหน้าเว็บ
    response = requests.get(url)

    # ใช้ BeautifulSoup เพื่อแยกโครงสร้าง HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # ค้นหาแท็ก <score-pairs-deprecated> เพื่อดึงข้อมูล
    scores = soup.find_all('score-pairs-deprecated')

    # เก็บข้อมูล criticsscore และชื่อหนัง
    criticsscores = []
    movienames = []
    for score in scores:
        criticsscore = score.get('criticsscore')
        moviename = score.find_next('span', class_='p--small').text
        if criticsscore:
            criticsscore = int(criticsscore)  # แปลงเป็นจำนวนเต็ม
            if criticsscore >= threshold:
                criticsscores.append(criticsscore)
                movienames.append(moviename)

    return criticsscores, movienames

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    # pip install bottleneck --upgrade
