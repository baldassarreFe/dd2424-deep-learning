# Deep Learning in Data Science - Assignment 4

##### Vanilla RNN trained with AdaGrad on a dataset of character sequences

## Gradient Checking

- Tried small and large networks
- Tried short and long sequences
- Tried small and large input/output sizes

## Goblet book

### graph
- note periodical high and lows corresponding to different parts of the romance
- longer training does not mean lower loss: in one run got 37 after 3 epochs,
in another one was above 40 after 20 epochs

### Training
- saving the weights every epochs for later generations
- generating every 3*4321 to avoid being always at the same point in the text
- here's up to 100000, more in the appendix

* Sequences 1 cost 111.008 elapsed 0s:
> kRM(;GfhHg.6^ H	CEpG)(lOoG(BvZd
yG!1GEMG-hGEe0mcGjtHjGHBGrXq,dgNWNO TsWQ)T!OG'IGd^TEYy)fAZ"C4G,4GH•GOjWX_AwYHtLPRG}DG6)G?BGY PmGHhGf
KE02z C6LI;tH4GIh
ZLKi GQ^:üYHEtG7GHBGK)GJ JcrP?GK•KH•GPoTycYOLOLbT

* Sequences 12964 cost 52.738 elapsed 70s:
> rs aor chare bo herede to and ivthem er ai them ha no hem.  wat and arof un tham has a feadd dovet wive goometevean qulmene, baldiin, bettaid.  SDowing of the st'd thim by apped in S bound to a porges

* Sequences 25927 cost 48.557 elapsed 145s:
>   nown'n sait Harry Masider but ling to eteed to snathe's dond.". ."
"I're warlen the sailedn'ld that at Harryf- hiund have - waivedye't beene was kring."
Fapream, what tore then of sabpleantwase tono

* Sequences 38890 cost 46.357 elapsed 218s:
> o the clace," Harry, sthice, lark he browh.
The rerets said wevery place yourlie fancy putting in eaded sting at bast.
Af he in pangny righer aroundf ped from one "Very a she the mase leed the "An't

* Sequences 48623 cost 51.954 elapsed 274s:
> tus.
"Auked ricak Cithing the.  Frok in thels any't.  Ther patless linged lonbs ard Gover didn't jugh in turing cuots Me?", hat Camtreching sherons off notlof, as here's Mrsly front whey.' sablis he b

* Sequences 61586 cost 48.828 elapsed 348s:
>  ofed whust deceonel, the on whbullaze inare wasds bet thevensss twere indit's, the winccingarne the eorden off ou bakecing leatilf nome get be, dow him inninter. WI dim disg welestorn.  Harry fifone

* Sequences 74549 cost 46.775 elapsed 423s:
> ng.
Ceed un as Harry nomwit onR dots start frot spend butgifh purusnione too look and Harry nearle thoutcriftidly wan op tlas well, brondised tcrild eyes surre butly gringlly.  gan thermbrullomaf here

* Sequences 87512 cost 45.047 elapsed 498s:
> th stastor soing in see hands.  He betnsuds looked of hiss upone guthing arled, to the he'ved tone his pook him.  Halle.  Hermy withen tolyed.  Is Word pert at .
He feen Muthor s. . . . . .  'moks tom

* Sequences 97245 cost 46.288 elapsed 556s:
> you s," sfo whaid thin seith themsay was suth!"
"Who sly and slightelle. ?"
"Deased.  "
"I frttiont I'll not than, bease's churne Why went tood indoup was -"E - liup the hip ued was atp wobling bake."

* Sequences 110208 cost 45.000 elapsed 630s:
> ded woulded at on time foth moll hal the lusilud hear har ole sire-thimpeyore sury sthe, - of he!" she woodher, lugge oum noth's a wheadil! ."
seasferor wotmest to cean bud.
"Yay nour. Youn reason you

### Generating text
- why warm up in generation when the net is just loaded and cold
- here are 1000 chars, up to 884930 more in the appendix

> "Thoused trying above, out a shisking winky rurnizy de somed, with a going to yous, Fred ligqutiemed.  Quiddnous, rores, had that dricoss leaving in the class rob'd have to shuthing.  "I dighind wha deady," shout, chair.  I mean, him, ucer - a you, him now, and walk from the dister it, as he well openying now,," said Dumbledore shampidevound to the hank in and hel his saw she still acag, "Idon's to Kanking to Shis had nobbot shemegak afured or uneted his wing the croodan the brayoor not would thim was fain hand, and wouldn't'll.  It was in frightly oigh of who might wan was quietly point, even an words.
>
> "You have firand in a shiday when the bomplipe very sleantifulsly.
>
> "That?"  Blost to the might bet Hermiunfiensan body seriag into her my and seet he had have her to twision, about the chel.
>
> Rent to nos?  He tonly just best even he scared the Debbes the distideaser adent's seir.  He pase.  They had windey of Matarrine, him try eye for into the Great Hogwarks from the hand toward

## Other books

- ulysses (more chars, longer -> harder)
- divina commedia (fewer chars, shorter -> easier)
