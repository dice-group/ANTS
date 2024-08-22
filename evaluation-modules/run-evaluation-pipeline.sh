#source ~/anaconda3/etc/profile.d/conda.sh
if [ -d "GenerationEval" ] 
then
    # overide GenerationEval/eval.py 
    cp eval.py GenerationEval
else

	# installing GenerationEval from WebNLG2020
	git clone https://github.com/WebNLG/GenerationEval.git

	cd GenerationEval

	# installing dependecies
	bash install_dependencies.sh
	pip install -r requirements.txt

	# download latest BLEURT checkpoint model
	cd metrics/bleurt
	wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
	unzip BLEURT-20.zip
	# back to GenerationEval folder
	cd ../..

	# back to evaluation modul folder
	cd ..
fi

# overide GenerationEval/eval.py 
cp eval.py GenerationEval

# back to root folder 
cd ..
