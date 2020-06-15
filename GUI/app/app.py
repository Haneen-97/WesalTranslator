import io
import json
from datetime import datetime
from functools import wraps
from io import BytesIO
import cv2
import numpy as np
from flask import json, send_from_directory, flash, session
from flask import request, redirect, render_template
from flask import url_for, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
from gtts import gTTS
from pygame import mixer

import frameExtraction
from app import app

# configure database
app.config['MYSQL_USER'] = 'sql12324934'
app.config['MYSQL_PASSWORD'] = 'XRMAkCRdug'
app.config['MYSQL_HOST'] = 'sql12.freemysqlhosting.net'
app.config['MYSQL_DB'] = 'sql12324934'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# initializes the database
mysql = MySQL(app)
CORS(app)


# homepage
@app.route("/index.html")
def index():
    return render_template('index.html')
# translation page
@app.route("/translation.html")
def translation():
    return render_template('translation.html')

@app.route("/")
def x():
    return render_template('index.html')


# admin login function
@app.route("/login.html", methods=["GET", "POST"])
def login():
    # the below codes to check if the login form submit using POST method or not
    if request.method == 'POST':
        # The bellow code to get the admin email and password
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        # The below codes to construct the connection with database and retrieve the admin email and password
        curl = mysql.connection.cursor()
        curl.execute("SELECT * FROM User WHERE username=%s", (email,))
        user = curl.fetchone()
        curl.close()
        # The below code are to check if there are information retrieved from the database
        if user != None:  # "> 0:
            # The below code are to check if the password correct or not
            if user["password"].encode('utf-8') == password:
                session['loggin'] = True # starts the session
                # If the password correct will redirect admin to the view log
                return redirect(url_for('view_log'))
            else:
                # If the password is incorrect will redirect admin to the log in page with an error message
                return render_template("login.html", errorM=" كلمة المرور خاطئة الرجاء ادخالها بطريقة صحيحة")
        else:
            # If the email is incorrect will redirect admin to the log in page with an error message
            return render_template("login.html", errorM=" البريد الالكتروني خاطئ الرجاء ادخاله بطريقة صحيحة")

    else:
        # To redirect users to Login page if they click " تسجيل الدخول" button
        return render_template("login.html")

    return render_template('login.html')


# admin logout function
@app.route("/logout.html")
def logout():
    session.pop('loggin', None)
    # To redirect users to main page if they click "الخروج تسجيل" button
    return render_template('index.html')


# model integration begins here
@app.route("/api/prepare", methods=["POST"])
def prepare():
    # get the requested file from the user
    file = request.files['file']
    # call preprocessing function that preprocess each image
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})


@app.route('/model')
def model():
    # app\model_js2\model.json
    # load the saved model in json format
    json_data = json.load(open("./model_js2/model.json"))
    # return model
    return jsonify(json_data)


# load shards of the model
@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js2', path)


# preprocessing images
def preprocessing(file):
    in_memory_file = io.BytesIO()
    # save the images in memory
    file.save(in_memory_file)
    # convert the image into numpy array
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    # convert the image into grayscale
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    # resize the image as the model input shape
    res = cv2.resize(img, dsize=(64, 64))
    return res

# implementing speech synthesizing once the user clicks on تشغيل it will be requested to this function
@app.route('/sendAudio', methods=["POST"])
def sendAudio():
    text = request.form['preview']  #Take the text from HTML
    tts = gTTS(text=text, lang='ar')
    #save the text on a variable, the language for the text is “ar”= arabic.
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp) #write the text on mp3 file
    mp3_fp.seek(0)
    mixer.init()  #Initialize mixer
    mixer.music.load(mp3_fp) #load the file
    mixer.music.play() #play the file
    return jsonify(preview=text)


# login function to check the session if it started or not
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loggin' in session:
            return f(*args, **kwargs)
        else:  # if the user is not logged in, will be redirected to the index page
            flash('الرجاء تسجيل الدخول', 'danger')
            return redirect(url_for('index'))
    return decorated_function

@app.route("/view_log.html", methods=["GET", "POST"])
@login_required
def view_log():
    # created a cursor to work with the database
    cur = mysql.connection.cursor()
    # when the method is post, the admin clicked the filter button
    if request.method == "POST":
        # when filter button clicked the filter request will be performed
        if request.form['filter'] == 'filterBtn':
            # store the 'datepicker' value in >>date<< variable
            # the date input format is (yyyy-mm-dd - yyyy-mm-dd)
            date = request.form['datepicker']
            # "split" divides the date variable on space and becomes an array of 3 elements
            date_array = date.split(" ")
            # from_date variable takes the from date yyyy-mm-dd >> index[0] <<
            from_date = date_array[0]
            # index[1] ignored
            # to_date variable takes the to date yyyy-mm-dd >> index[2] <<
            to_date = date_array[2]
            # selects the logs with success status
            cur.execute("SELECT * FROM Logs WHERE status = %s AND created_at BETWEEN  %s AND %s",
                        ('success', from_date, to_date))
            # counts the number of success statuses
            var1 = cur.rowcount
            mysql.connection.commit()
            # selects the logs with failure status
            cur.execute("SELECT * FROM Logs WHERE status = %s AND created_at BETWEEN  %s AND %s",
                        ('failure', from_date, to_date))
            # counts the number of failure statuses
            var2 = cur.rowcount
            mysql.connection.commit()
            # the return statements render the page and sent the count values of success and failure status
            return render_template('view_log.html', success=var1, failure=var2)
    # whenever the user access the page it will implements this part
    else:
        # selects the logs with success status
        cur.execute("SELECT * FROM Logs WHERE status = 'success'")
        # counts the number of success statuses
        var1 = cur.rowcount
        mysql.connection.commit()
        # selects the logs with failure status
        cur.execute("SELECT * FROM Logs WHERE status = 'failure'")
        # counts the number of failure statuses
        var2 = cur.rowcount
        mysql.connection.commit()
        # the return statements render the page and sent the count values of success and failure status
        return render_template('view_log.html', success=var1, failure=var2)


@app.route("/modalAnswer", methods=["POST"])
def modalAnswer():
    # created a cursor to work with the database
    cur = mysql.connection.cursor()
    if request.method == "POST":
        # put the current date (when the user clicked the button) in now variable
        now = datetime.now()
        # change the date format
        formatted_date = now.strftime('%Y-%m-%d')
        # when the user clicks the button with 'success' value enters the if statement
        if request.form['correct'] == 'success':
            # insert a new log with success status and the formated_date
            cur.execute("INSERT INTO Logs (status,created_at) values (%s,%s)", ('success', formatted_date))
            mysql.connection.commit()
        # when the user clicks the button with 'failure' value enters the elif statement
        elif request.form['correct'] == 'failure':
            # insert a new log with failure status and the formated_date
            cur.execute("INSERT INTO Logs (status,created_at) values (%s,%s)", ('failure', formatted_date))
            mysql.connection.commit()
        return render_template('translation.html')

import image
@app.route('/about', methods=["POST"])
def operate():
    if request.form['pass'] == "videoPressed":
        frameExtraction.videoProcessing()
        # os.chdir("./app")
        # print(os.chdir("./app"))
        # print(os.chdir(".."))
        return render_template('translation.html')
    elif request.form['pass'] == "imagePressed":
        image.uploadImage()
        # os.chdir("..")
        return render_template('translation.html')
    else:
        return render_template('translation.html')
    #     frameExtraction.videoProcessing()
    # frameExtraction.cube()
    # else:
    #     return render_template("view_log.html")
    # videoProcessing()



if __name__ == '__main__':
    app.run(debug=True)
