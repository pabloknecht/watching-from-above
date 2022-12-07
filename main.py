
from wfa.outils.get_new_images import get_s2maps_data

if __name__ == "__main__":
   print("File one executed when ran directly")
   print(get_s2maps_data(6.4,48.903112, 2.195586,'2019'))
