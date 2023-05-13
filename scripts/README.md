### Train classifier to predict win v.s (loss or draw)

A breakdown of the variables to train with:

* poss = possesion (% of passes attempted)
* xg   = expected goals
* touches = number of touches
* def 3rd = number of touches in def 3rd
* mid 3rd = similar to above
* att 3rd = similar to above
* att pen = similiar to above
* live = live-ball touches
* ast = assists
* xag = expected assisted goals
* xA = expected assists
* KP = key passes
* 1/3 = passes into final third
* ppa = passes into penalty area
* crspa = crosses into penalty area
* prgp = progressive/forward passes 
* npxg = non-penalty expected goals
* npxg/sh = same as above but per shot
* g-xg = goals minus expected goals
* npg-npxg = non pen goals minus non pen expected goals
* team = one hot team names 
* hours = closest hour of the day match was played in
* day = day of week match was played on
* opponent = one hot team names
* gf = goals for
* ga = goals against
* formation = one hot formations
