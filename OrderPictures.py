import glob, os, shutil

# Settings
# -------------------------------------

pictureMainFolder = "extracted_images"

folderMain = "dataset"
folder1Name = "Training"
folder2Name = "Testing"
folder3Name = "I_Forgot"

# Percent of images to be placed per folder
folder1Number = 0.7
folder2Number = 0.15
folder3Number = 0.15

foldersToGrab = ["1", "!", "sigma"]
# -------------------------------------


# Code
# -------------------------------------

if not os.path.exists(folderMain):
    os.mkdir(folderMain)
if not os.path.exists(folderMain + "/" + folder1Name):
    os.mkdir(folderMain + "/" + folder1Name)
if not os.path.exists(folderMain + "/" + folder2Name):
    os.mkdir(folderMain + "/" + folder2Name)
if not os.path.exists(folderMain + "/" + folder3Name):
    os.mkdir(folderMain + "/" + folder3Name)

for folder in foldersToGrab:
    printing = 0
    print("Copying folder " + folder)

    images = glob.glob(pictureMainFolder + "/" + folder + "/*.jpg")
    temp1Num = int(len(images) * folder1Number)
    temp2Num = int(len(images) * folder2Number)
    temp3Num = int(len(images) * folder3Number)

    # copies and renames files
    for image in images:
        if temp1Num > 0:
            shutil.copy(image, folderMain + "/" + folder1Name + "/" + folder + str(temp1Num) + ".jpg")
            temp1Num -= 1
            if printing == 0:
                print("Copying to first folder")
                printing += 1
        elif temp2Num > 0:
            shutil.copy(image, folderMain + "/" + folder2Name + "/" + folder + str(temp2Num) + ".jpg")
            temp2Num -= 1
            if printing == 1:
                print("Copying to second folder")
                printing += 1
        elif temp3Num > 0:
            shutil.copy(image, folderMain + "/" + folder3Name + "/" + folder + str(temp3Num) + ".jpg")
            temp3Num -= 1
            if printing == 2:
                print("Copying to third folder")
                printing += 1
    print("Copying finished for folder " + folder + "\n")

print("Images copied")

# -------------------------------------
