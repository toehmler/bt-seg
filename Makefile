update:
	git add .
	git commit -m 'quick fix'
	git push

ptest:
	python3 Scripts/patch_test.py



predict: 
	python3 Scripts/predict.py

pre:
	python3 Scripts/preprocess.py

train:
	python3 Scripts/train.py

test: 
	python3 Scripts/test.py


