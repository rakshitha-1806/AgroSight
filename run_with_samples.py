
from app import app
from flask import render_template
@app.context_processor
def inject_sample_images():
    return dict(sample_images=['Tomato_Early_blight_0.png', 'Tomato_Early_blight_1.png', 'Tomato_Early_blight_2.png', 'Tomato_Early_blight_3.png', 'Tomato_Healthy_0.png', 'Tomato_Healthy_1.png'])

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
