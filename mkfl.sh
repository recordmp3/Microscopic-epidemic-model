#rm *.o
#rm main
#g++ -c city.cpp -std=c++11
#g++ -c config.cpp -std=c++11
#g++ -c environment.cpp -std=c++11
#g++ -c facility.cpp -std=c++11 
#g++ -c individual.cpp -std=c++11
#g++ -c main.cpp -std=c++11
rm results/*/files/simu*.txt
rm results/*/files/debug*.txt
rm results/*/files/*.so
rm -r results/*/files/build
python3 build.py build_ext --inplace
#g++ config.cpp city.cpp environment.cpp facility.cpp individual.cpp main.cpp -o main
python3 main.py

