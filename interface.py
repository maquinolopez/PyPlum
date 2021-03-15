rom tkinter import *
from tkinter.ttk import *

#Dibuja la ventana principal
root=Tk()

#título y tamaño de la ventana
root.title("Interfaz para PLUM")
root.geometry('600x400')

#texto que te va aparecer (no se dibuja hasta que hacemos label.pack() abajo)
label=Label(root,text="Introduzca la ubicación de su archivo de datos en csv")

#cajita de texto donde pueden poner la ubicación de su archivo
archivo = Entry(root)

def ledieronclick():
    #Aquí le dices que corra PLUM.
    #Puedes imprimir cuando quieras con comandos como este:
    mensaje=Label(root,text = "el archivo elegido fue " + archivo.get())
    mensaje.pack()
    
    #También puedes cambiar propiedades de cosas que ya están ahí
    label.configure(text="Espere a que termine de correr PLU")

btn=Button(root,text='Correr PLUM!', command = ledieronclick)

label.pack()
archivo.pack()
btn.pack()
root.mainloop()



