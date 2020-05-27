import glob
import os
import pickle
import librosa
import numpy as np
import soundfile
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_feature(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

ravdees_emotions={
    "01":"neutral",
    "02":"calm",
    "03":"happy",
    "04":"sad",
    "05":"angry",
    "06":"fearful",
    "07":"disgust",
    "08":"surprised"
}

available_emotions={"angry","sad","happy"}
y=[]
x=[]
for file in glob.glob("E:/ravdees/data/Actor_*/*.wav"):
    basename=os.path.basename(file)
    emotionno=basename[6:8]
    emotion=ravdees_emotions[emotionno]

    if emotion not in available_emotions:
        continue
    features = extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(features)
    y.append(emotion)


X_train, X_test, y_train, y_test =train_test_split(np.array(x), y, test_size=0.2, random_state=0)
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500,
}
# initialize Multi Layer Perceptron classifier
model = MLPClassifier(**model_params)
# train the model
print(" ..........Training the model........")
model.fit(X_train, y_train)
print("!!!training completed!!!")

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
#print(os.path)

# save the model
# make result directory if doesn't exist
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))

 # load the saved model (after training)
model = pickle.load(open("result/mlp_classifier.model", "rb"))
filename="E:/test/angry1.wav"
features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1, -1)
# predict
#result = model.predict(features)

 # show the result !
#print("result:", result[0])