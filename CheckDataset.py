import os

INPUT_DIR = r"D:\Project Folder\Mini Project\Datasets\Audio Datasets\Raw Dataset"

classes = ["AI Voice", "Human Voice"]

for class_name in classes:

    folder = os.path.join(INPUT_DIR, class_name)

    wav = 0
    mp3 = 0
    flac = 0
    other = 0

    for file in os.listdir(folder):

        extension = os.path.splitext(file)[1].lower()

        if extension == ".wav":
            wav += 1

        elif extension == ".mp3":
            mp3 += 1

        elif extension == ".flac":
            flac += 1

        else:
            other += 1

    print("\n==========================")
    print(class_name)
    print("==========================")
    print("WAV   :", wav)
    print("MP3   :", mp3)
    print("FLAC  :", flac)
    print("Other :", other)
    print("Total :", wav + mp3 + flac + other)