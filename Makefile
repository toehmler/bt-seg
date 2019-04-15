update:
	git add .
	git commit -m 'quick fix'
	git push

pre:
	python3 Scripts/preprocess.py

train:
	python3 Scripts/train.py

test: 
	python3 Scripts/test.py


