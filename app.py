from flask import Flask, request, redirect, url_for,render_template,send_from_directory
from werkzeug.utils import secure_filename
import os
from app.mask_detector import *
from app.face_recognition import face_recognition_api
from app.face_catch import face_catch_api
app = Flask(__name__)


#from app.mask_detector import mask_detector_api


#app.register_blueprint(mask_detector_api,url_prefix='/') 

num=0
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



################
#show upload image
################
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

################
#upload mask image
################
@app.route('/', methods=['GET', 'POST'])
def home_page():
    return  render_template("home.html")

@app.route('/mask', methods=['GET', 'POST'])
def maskornot():
    global num
    if request.method == 'POST':
        # check if the post request has the file part
        #if 'file' not in request.files:
            #flash('No file part')
            #return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        #if file.filename == '':
        #    flash('No selected file')
        #    return redirect(request.url)
        if file and allowed_file(file.filename):

            
            filename = secure_filename(file.filename)
            #filename= str(num) +".jpg"
            #num=num+1
            #if num==100:
                #num=0

            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',filename=filename))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            mask_detect_showimg(file.filename.split('.')[0],filename)
            predict_filenamae = './static/predict_'+file.filename

            

            return render_template("mask.html", test_image = filename ,predict_image = predict_filenamae)
            
    return  render_template("mask.html")


################
#upload kim or trump face image
################
@app.route('/certain_person', methods=['GET', 'POST'])
def certain_person_recognition():
    if request.method == 'POST':
        # check if the post request has the file part
        #if 'file' not in request.files:
            #flash('No file part')
            #return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        #if file.filename == '':
        #    flash('No selected file')
        #    return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',filename=filename))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            res = face_recognition_api(filename)
            print(res)
            return render_template("certain_person.html",test_image = filename,res = res)
            
    return  render_template("certain_person.html")


################
#upload kim or trump image
################
@app.route('/facecapture', methods=['GET', 'POST'])
def facecapture():
    if request.method == 'POST':
        # check if the post request has the file part
        #if 'file' not in request.files:
            #flash('No file part')
            #return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        #if file.filename == '':
        #    flash('No selected file')
        #    return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',filename=filename))
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            res = face_catch_api(filename)
            return render_template("facecapture.html",test_image = filename,res = res)
            
    return  render_template("facecapture.html")



#美化用
#
# @app.route('/face',methods=['GET'])
# def face_recognition():
#     face_image_list = ['static/kim/'+i for i in os.listdir('static/kim')]
#     face_image_list += ['static/trump/'+i for i in os.listdir('static/trump')]
#     return render_template("face.html", len = len(face_image_list),face_image=face_image_list)

# @app.route('/face_check/<filename>',methods=['GET','POST'])
# def face_check(filename):
#     face_image_list = ['static/kim/'+i for i in os.listdir('static/kim')]
#     face_image_list += ['static/trump/'+i for i in os.listdir('static/trump')]
#     res = face_recognition_api(filename)
#     print(res)
#     return render_template("face.html", len = len(face_image_list),face_image=face_image_list)




if __name__ == '__main__': 
    app.run('0.0.0.0',debug=True)