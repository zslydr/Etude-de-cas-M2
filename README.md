# Etude de cas M2

L'objet initial du projet est de pouvoir répondre à un cas d'application concernant l'usage de caméras embarquées en voiture (pour par exemple assister les voitures automatiques). L'idée est ici de construire dans un premier temps un modèle de classification pour pouvoir identifier/classifier correctement des panneaux de signalisation. La recherche dans le traitement de l'image à largement était propulsée récemment par les réseaux de neuronnes convolutifs (grâce notamment aux avancées en puissance de calculs graphiques que l'on doit à nos amis les gamers).

Dans un second temps, nous aimerions nous interesser à des cas dégénératifs, lorsque l'information obtenue par la caméra est dégradée (par exemple si un panneaux est tagué). Contrôler le comportement du réseau de neuronnes : Est-il en meusure de toujours bien classifier les panneaux, ou cela fausse-t-il les performances de ce dernier ? Le cas échéant peut-on meusurer cette perte de performance ?
