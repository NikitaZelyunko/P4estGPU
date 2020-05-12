git submodule init && git submodule update
cd p4est
git submodule init && git submodule update
./bootstrap
./configure --enable-mpi --enable-openmp
# при вызове configure стоит использовать --prefix=path/to/build
# например создать папку которую потом указать в корневом CMakeLists
make && make install

I. Необходимые программы
sudo apt-get install gfortran
sudo apt-get install mpich
sudo apt install zlib1g
sudo apt-get install libreadline-dev

II. Необходимые библиотеки которые нужно установить из исходников

Lapack & Blas
https://ahmadzareei.github.io/azareei/linux/2016/04/08/configuring-blas-lapack.html
1. Скачать Blas с официального сайта http://www.netlib.org/blas/
2. Распаковать Blas в папку и перейти в неё
gfortran -O3 -c *.f # compiling
ar cr libblas.a *.o # creates libblas.a
sudo cp ./libblas.a /usr/local/lib/ 

3. Скачать Lapack с официального сайта http://www.netlib.org/lapack/
3.1 Лучше sudo apt-get install liblapack-dev
4. Распаковать Lapack в папку и перейти в неё
5. В файле make.inc.example изменить значение BLASLIB вот так:
BLASLIB = /usr/local/libblas.a
и сохранить этот файл как make.inc
sudo make
Если появится ошибка 
Makefile:463: recipe for target 'znep.out' failed
выполнить команду
ulimit -s unlimited
После завершения сборки
sudo cp ./liblapack.a /usr/local/lib/

6. Скачать Zlib с официального сайта http://www.zlib.net/
7. Распаковать Zlib в папку и перейти в неё
./configure
sudo make install

8. Скачать Lua с официального сайта http://www.lua.org/download.html
9. Распаковать Lua в папку и перейти в неё
make linux test
Если будет ошибка сделать sudo apt-get install libreadline-dev
sudo make install

10. Скачать драйвер для видеокарты
11. Скачать CUDA-toolkit

