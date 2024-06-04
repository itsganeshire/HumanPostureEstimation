import threading
from flask import Flask,render_template, request
import matplotlib.pyplot as plt
from pandas import read_csv

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/info2')
def info2():
    return render_template('info2.html')

@app.route('/info3')
def info3():
    return render_template('info3.html')

@app.route('/lunges',methods=['POST','GET'])
def lunges():
    count=request.form.get('count')
    if count is None:
        count=0
    else:
        count=int(count)
        
    if isinstance(count,int):
        from HumanPosture import lunges as s
        from HumanPosture import voicemodule as v
        s(count)
        
        t1=threading.Thread(target=s)
        t2=threading.Thread(target=v)
        
        t1.start()
        t2.start()
    return render_template('lunges.html')

@app.route('/plank')
def plank():
    return render_template('plank.html')

@app.route('/squat',methods=['POST','GET'])
def squat():
    count=request.form.get('count')
    if count is None:
        count=0
    else:
        count=int(count)
        
    if isinstance(count,int):
        from HumanPosture import squat as s
        from HumanPosture import voicemodule as v
        s(count)
        
        t1=threading.Thread(target=s)
        t2=threading.Thread(target=v)
        
        t1.start()
        t2.start()
        
    return render_template('squat.html')

#@app.route("/squatresult")
# def sresult():
#     df=read_csv('data.csv')
#     l=list(df['counter'])
    
#     ecount=(l.count('ERROR')/len(l))*100
#     rcount=100-ecount

#     res=[]
#     res.append(ecount)
#     res.append(rcount)

#     plt.figure("Result")
#     plt.title('Result')
#     plt.pie(res,labels=['Error','Correct'],colors=['r','g'])
#     plt.show()

#     return render_template('squat.html')
@app.route('/result')
def google_pie_chart():
    df=read_csv('data.csv')
    l=list(df['counter'])
    
    ecount=int((l.count('ERROR')/len(l))*100)
    rcount=100-ecount
    data = {'Task' : 'Hours per Day','Correct':rcount,'Error':ecount}
    return render_template('result.html', data=data)
# @app.route("/result",methods=['POST','GET'])
# def result():
#     print(request.form.get('name'))
#     return render_template('temp.html')

if __name__== '__main__':
    app.run(debug=True)