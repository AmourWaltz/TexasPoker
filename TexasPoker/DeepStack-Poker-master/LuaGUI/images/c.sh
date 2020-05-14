for file in *.png
    do convert -quality 100 $file ${file%%.*}.bmp
done
