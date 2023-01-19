DATASETS="Art Books Dolls Laundry Moebius Reindeer"
for NAME in ${DATASETS}; do
    mkdir -p ${NAME}
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/disp1.png -O ${NAME}/disp1.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/disp5.png -O ${NAME}/disp5.png;
    wget -nc https://vision.middlebury.edu/stereo/data/scenes2005/FullSize/${NAME}/dmin.txt -O ${NAME}/dmin.txt;

done
