# CharRNN

## Usage

### Train on HP

From scratch

```bash
python3 -OO train.py -s 100 sources/goblet_book.txt
```

From a checkpoint

```bash
python3 -OO train.py -c weights/goblet.npz sources/goblet_book.txt
```

### Generate from HP

```bash
python3 -OO generate.py -l 1000 sources/goblet_book.txt weights/goblet.npz
```

## Testing
```bash
python3 -m unittest
```

## Results

### Goblet

### Ulysses

Training on sources/ulysses.txt:
- 1534227 total characters
- 123 unique characters
- 61369 sequences of length 25

After 675059 iterations (11 epochs):

![Ulysses](plots/ulysses.png)

```txt
The scoming moll on the corcennt allorgadny!

God Gracking of thoukergect in fit and ereertbits and all a kit.

Wom it knieper shurgementwatewent wot?

For then
vartenquitibininct to brow alioniins gluppajests damited notenies one revit or one ourd drstalle mromuminf it bordatioo a
wead,
it’s revite the andores wetrelise to the mocket more of Greeco vecteng
‘liwning inomEnsed now casternion 1. We too luccivebard, O. He gapens
sare the out Slace with the
croage and arcucat Cot trowoly by Gutting beasfbount
bo interments agife in tidy
vodeting about the pauthy all casess o have o. 11 drave of with cankting Pallad ockewe.

  O0 frayly cover Maice the with
7/9 By oHs the wheme was on im to
forcher net from the herrlengyring was you allation ot
wink Fougusly old decumproushing in Emwand on
imelly thing withor beled sherwycad prautch wrump unfoub him was not dook and a with casso!

Viceal on immatues to beto whine a pomacegrenting of a nawed avelranyly—cong benett of Illig two vorarco acht You or Guth
he praum vire.

Him I done tward treps
courters becakead father
Hy of pracest Araepy
the
Wes dram his fealway outer
shorter accanist Of they olingly I from, such
lows wis depupt Prop
excomase cankers pricnang comshor whohe-there G...‘ flouse
well
not! Or co-trimesced theres
it Panition invers of it rech gealing to lane. The vargegd of Incricion com all joont antters goor of ligulate ere dut by 170 Ladnion, mutter inkerkske as tupting addelks of a ciposhart aublen to gunice belual accuoss deys Rea their
blituly over the 2 rece tymlounsents
or but to hadgetrmedaring
in the dal proculent and any invoseld
my, was yoting with comboom dogobibmenevam
over you its astion a
me pronestive bicked alougss coodupates, entitiding yes ut in ereay at with
thourse I parged triming showor so 1.
Alf mure the sitrelf
Greewchation I oentide other. Jafficent ant they kibrictiong of wouncations of 5 Im king be sumpo the vots reodesose Giverge
the
Gllopprett
no juage of elching suncitulays a prys of over in the fredace over to him bliimbote overaymated banks yelpitapolido, Douning Stejung but abo, poitlow, How comperser of mingatt by mondroned I loves this forgry prabed who git duakd as the
sun crrige boing Coneraring ould deever kiny in owifader Is connefes and or orger mild
gondouly onugited croniamay poure a bibour my ase hosking sprazing for the nompion of ore he warigice they ought itman argingh by it
her tior thaisents he go me 1? 30 the stear moned time everonanateyd her landure on the crall to nod Is the eres
a sheud pare clice of
ent petionding for Gady thtire old but thons thouthne of him me frjeveval
to mut it of the waimbled mists Mally nightargefle
in suynlox sis andor he gos as all Wanderss deseraned wark I
luck
for the 6% 3me yerner imientatcant Comer be whith losal welle wall beoughorks auses ostcers of high sontematides of mugnicavest sa
sisiatid and would yound brats Lit him apical the
```

### Divina Commedia

Training on sources/divina_commedia.txt:
- 547990 total characters
- 81 unique characters
- 21919 sequences of length 25

After 219190 iterations (10 epochs):

![Divina Commedia](plots/divina_commedia.png)

```txt
che più tascorva ettro fenca, somma;
e di l'al peranno e la laghi ussitte meti sé dal cardo li nastamo vittar che viuttita piele è ma eruto l'ua incin, dissi
morlan, pappe acceente sua più fesi,
pur corsima ammi così che disse,
e mora lu ita,
che tu velumed panto,
ché prima sen tuspuora ia vie il povranti».
Di Vemo, vicchia tu molineli a Dice, privesture,
ché l'agra a indi contai coter parti malsu mante al si mi tuo verta
ond' inché diossegne.
Io
cer ellrando,
mi santa mento
torco che la grondo la fiscusfia
che in vigri,
Maluna voi
dïuole a fui e dignae vande
che di quei de da guardo
codambrisa.
Tumi sua pinse 'n coleso, convidella,
che tura che vidi in la possa, lifamesite l'uefiol par stisso più suoce li sper che di ch'io sobra!",
ne l'asp'ata e ontipro fai che diesti, penconne
XVe eltrò fiamette, fe' mi del pritto, di me acero
ento ad prei etter, rante,
non d'a, farglisa ter perolio
Lopirmi, o grirciò no puttri ove non che con lui frimor fenso,
quella vi la vonne;
che mal siol amprenzi in chentamadette l'ulla divompute
lo narei più piusa.]
 lì moi d'acai nen suari chessi iffarele,
incorüa
ponto di t'avaga, cun tior le usto amote piei
ch'ardi»,
lo mio e sé per come Parne;
conferito, sespondo unvi vedime più ch'andolizi' io il sossi credendaco più l'altre
si assudi,
non pun che 'l diso rre volle,
si tiati,
ferra».
.]
 nal lune
al soto
dil sovo
pervento il me di düa me più nel bettirile il paco ignoi:
con fiaite.
E 'h Bëal quella, mi disolte parela».
TErila 'n la divendo i fianta; ondo fieni che intra a lorello oScun legnon sì che pari,
ch'in che la tustra dia fiar suo quando inrio sattel tonvanza tre d'altre, che di refre di colse
cin diverna merder dissute perta di que caenza, megliegrà di CArere e si dolge rapente mempre
giuo del Magiongier quinta, pur son Bear benir che vissinïara
saltamerta nette in prissa è vodi è l'inoa come catù di colle sentresti, qua di edco di bestilutti
e giè le polte;
lui tal mia che mio per Malto a tal meste;
perinandi
di ranto, contri artala illando ferai che tu si civi, mense verne,
distutto, scariti;
ra penze in quesce Cil gueneorino giuste le screngegita, di, questa nento de la pol venti e tarsole;
cersteer li guar de l'alternte pacegno to che non tui
compe amtricera.
Fonde Daro,
sur proscattigrimo retta di dostirla e dinutre li maturna guenta.
Sì diane sela la fétro n'uso, non qua visi tirola XXII, volcia me: «Sì fa tu diso.
Di mi bëa velizzio s'ascan tervo;
a lui p'he adietto a per antascerci vesse ondi i azzia!
Qui la pnel sonta e per sece di l'ie, porate".
Egre avierì riotte.
O in lui di vi puù tua noglio a la coma di s'altra
rïatù che prota è tiendamingriva,
pervinutta vichi a destro, e fonte
di guartïante da priceadarala inchi per li credo,
quesno
venime songarmo
che termando,
inde ia fondore che de l'emer fé sobtese in la sertia somenessazeledi sodi pira a fistode de la percunai.
O dossume
fionna,
e sonvesi per peamar priccinuma convazzorgò s'asche,
che lisadi nedo di Vose al suo per che e a pere al tisse
```
