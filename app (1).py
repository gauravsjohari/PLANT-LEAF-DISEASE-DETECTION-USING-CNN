from flask import Flask, request, render_template,url_for
import os
from werkzeug.utils import secure_filename

import tensorflow as tf 
import numpy as np
from keras.preprocessing import image
import dbConnection as db

app = Flask(__name__,template_folder='template')
app.config['UPLOAD_PATH'] = os.path.join(os.getcwd(),'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
@app.route('/index',methods=['GET','POST'])
def index():
    return render_template('/index.html')    


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['imageInput']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            classifierLoad = tf.keras.models.load_model('model.h5')
            resultData=""
            Message=""
            IMGpath= os.path.join(app.config['UPLOAD_PATH'],filename)
            test_image = image.load_img(IMGpath, target_size = (200,200))
            #test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifierLoad.predict(test_image)
            if result[0][1] == 1: 
                resultData = "Potato___Early_blight" 
                result = 1
                db.InsertInto(filename,resultData)
                return render_template('/index.html', resultValue = result, fileName = filename, result1=resultData)
      
        
            elif result[0][0] == 1:
                resultData="Potato___healthy"
                result = 2
                db.InsertInto(filename,resultData)
                return render_template('/index.html', resultValue = result, fileName = filename, result1=resultData)
      
        
        
    

            elif result[0][2] == 1:
                resultData="Potato___Late_blight"
                result = 3
                db.InsertInto(filename,resultData)
                return render_template('/index.html', resultValue = result, fileName = filename, result1=resultData)
      
      
    return render_template('/index.html', fileName = filename, result1=resultData)
      
        
        

if __name__ == '__main__':  
    app.run(debug=True)
