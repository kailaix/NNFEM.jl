if [ ! -d "Adept-2" ]; then
  git clone https://github.com/rjhogan/Adept-2
fi
cd Adept-2
autoreconf -i
./configure
make
make check
make install