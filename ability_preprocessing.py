import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast  # for safely evaluating string representation of lists

# Read the dataset
df = pd.read_csv('all_mtg_cards_cleaned_image.csv', low_memory=False)

df = df.drop_duplicates(subset="name", keep="first")

df["attach"] = (df.text.str.contains("attach"))

## counterspells
df["counterspell"] = (df.text.str.contains("[Cc]ounter\\s(?:it|target|all)") & # find counterspells
         (df.text.str.contains("[wW]ard(?:\\s{|—])") == False)) # leave out ward cards

## exile
df["exile"] = (df.text.str.contains("[eE]xile\\s(?:target|each|all|the|up\\sto)") & # find exile
        (df.text.str.contains("the\\stop") == False)) # leave out impulse draw

## fight
df["fight"] = (df.text.str.contains("fights"))

## mill
df["mill"] = (df.text.str.contains("[mM]ill"))

## sacrifice
df["sacrifice"] = (df.text.str.contains("[sS]acrifice"))

## scry
df["scry"] = (df.text.str.contains("[sS]cry"))

## tap other
df["tap"] = (df.text.str.contains("(?:\\st|T)ap\\s(?:it|target|each|all|or\\suntap)") | # find active tappers
         df.text.str.contains("enter\\sthe\\sbattlefield\\stapped")) # as well as passive ones

## untap other
df["untap"] = (df.text.str.contains("[uU]ntap\\s(?:it|target|each|all)")) # find untappers

## deathtouch
df.loc[df.text.str.contains("[dD]eathtouch") | # find creatures that have deathtouch
        df.text.str.contains("deals combat damage to a creature, destroy that creature", regex = False)] # or that have "derptouch"

## defender
df["defender"] = (df.text.str.contains("[dD]efender") & # find creatures with defender or things that give defender
        (df.text.str.contains("(?:[tT]arget|[eE]ach|[aA]ll)\\screature(?:\\s|s\\s)with\\sdefender") == False)) # remove things that specifically affect defenders

## double_strike
df["double_strike"] = (df.text.str.contains("[dD]ouble\\sstrike"))

## first_strike
df["first_strike"] = (df.text.str.contains("[fF]irst\\sstrike"))

## flash
df["flash"] = (df.text.str.contains("(?:f|\nF|^F)lash") & # some engineering to avoid incorrectly grabbing cards with Flash in the name
        (df.text.str.contains("[fF]lashback") == False)) # dont' want to capture flashback

## flying
df["flying"] = (df.text.str.contains("[fF]lying"))

## haste
df["haste"] = (df.text.str.contains("[hH]aste"))

## hexproof
df["hexproof"] = (df.text.str.contains("[hH]exproof"))

## indestructible
df["indestructible"] = (df.text.str.contains("[iI]ndestructible") &
                         (df.text.str.contains("loses\\sindestructible") == False))

## lifelink
df["lifelink"] = (df.text.str.contains("[lL]ifelink"))

## menace
df["menace"] = (df.text.str.contains("[mM]enace"))

## protection
df["protection"] = (df.text.str.contains("[pP]rotection\\sfrom"))

## prowess
df["prowess"] = (df.text.str.contains("[pP]rowess"))

## reach
df["reach"] = (df.text.str.contains("(?:\\sr|\nR|^R)each") &
        (df.text.str.contains("can't be blocked except by creatures with flying or reach", regex = False) == False)) # don't want flying reminder text

## trample
df["trample"] = (df.text.str.contains("[tT]rample"))

## vigilance
df["vigilance"] = (df.text.str.contains("[vV]igilance"))

## draw
df["draw"] = (df.text.str.contains("(?:\\sd|\nD|^D)raw"))

## discard
df["discard"] = (df.text.str.contains("[dD]iscard"))

## damage
df["damage"] = (df.text.str.contains("deals\\s\\d\\sdamage"))

## damage prevention
df["damage_prevention"] = (df.text.str.contains("[pP]revent\\s"))

## life_gain
df["life_gain"] = (df.text.str.contains("gain(?:\\s|s\\s)\\d+\\slife"))

## life_loss
df["life_loss"] = (df.text.str.contains("loses") & 
                   df.text.str.contains("(?:their|\\d+)\\slife")) # capture both fixed and rational values

## tokens
df["tokens"] = (df.text.str.contains("[cC]reate"))

## destroy
df["destroy"] = (df.text.str.contains("[dD]estroy") &
                  (df.text.str.contains("don't\\sdestroy\\sit.") == False)) # reject indestructible's reminder text

## bounce
df["bounce"] = (df.text.str.contains("[rR]eturn") &
        df.text.str.contains("owner's\\s(?:hand|library)") & # capture hand or library bounce effects
        (df.text.str.contains("graveyard\\sto") == False)) # exclude grave recursion

## recursion
df["recursion"] = (df.text.str.contains("\\sput|return") &
        df.text.str.contains("graveyard")&
        df.text.str.contains("hand|battlefield"))

# Set specific keywords

df["absorb"] = (df.text.str.contains("[aA]bsorb"))
df["adapt"] = (df.text.str.contains("[aA]dapt"))
df["affinity"] = (df.text.str.contains("[aA]ffinity"))
df["afterlife"] = (df.text.str.contains("[aA]fterlife"))
df["aftermath"] = (df.text.str.contains("[aA]ftermath"))
df["amplify"] = (df.text.str.contains("[aA]mplify"))
df["annihilator"] = (df.text.str.contains("[aA]nnihilator"))
df["ascend"] = (df.text.str.contains("[aA]scend"))
df["bestow"] = (df.text.str.contains("[bB]estow"))
df["bolster"] = (df.text.str.contains("[bB]olster"))
df["bloodthirst"] = (df.text.str.contains("[bB]loodthirst"))
df["bushido"] = (df.text.str.contains("[bB]ushido"))
df["buyback"] = (df.text.str.contains("[bB]uyback"))
df["cascade"] = (df.text.str.contains("[cC]ascade"))
df["champion"] = (df.text.str.contains("[cC]hampion"))
df["changeling"] = (df.text.str.contains("[cC]hangeling"))
df["cipher"] = (df.text.str.contains("[cC]ipher"))
df["clash"] = (df.text.str.contains("[cC]lash"))
df["conspire"] = (df.text.str.contains("[cC]onspire"))
df["convoke"] = (df.text.str.contains("[cC]onvoke"))
df["crew"] = (df.text.str.contains("[cC]rew"))
df["cycling"] = (df.text.str.contains("[cC]ycling"))
df["dash"] = (df.text.str.contains("[dD]ash"))
df["delve"] = (df.text.str.contains("[dD]elve"))
df["detain"] = (df.text.str.contains("[dD]etain"))
df["devour"] = (df.text.str.contains("[dD]evour"))
df["dredge"] = (df.text.str.contains("[dD]redge"))
df["echo"] = (df.text.str.contains("[eE]cho"))
df["embalm"] = (df.text.str.contains("[eE]mbalm"))
df["emerge"] = (df.text.str.contains("[eE]merge"))
df["entwine"] = (df.text.str.contains("[eE]ntwine"))
df["epic"] = (df.text.str.contains("[eE]pic"))
df["evolve"] = (df.text.str.contains("[eE]volve"))
df["evoke"] = (df.text.str.contains("[eE]voke"))
df["exalted"] = (df.text.str.contains("[eE]xalted"))
df["exert"] = (df.text.str.contains("[eE]xert"))
df["exploit"] = (df.text.str.contains("[eE]xploit"))
df["explore"] = (df.text.str.contains("[eE]xplore"))
df["extort"] = (df.text.str.contains("[eE]xtort"))
df["fabricate"] = (df.text.str.contains("[fF]abricate"))
df["fading"] = (df.text.str.contains("[fF]ading"))
df["fateseal"] = (df.text.str.contains("[fF]ateseal"))
df["flanking"] = (df.text.str.contains("[fF]lanking"))
df["flashback"] = (df.text.str.contains("[fF]lashback"))
df["flip"] = (df.text.str.contains("[fF]lip"))
df["forecast"] = (df.text.str.contains("[fF]orecast"))
df["foretell"] = (df.text.str.contains("[fF]oretell"))
df["fortify"] = (df.text.str.contains("[fF]ortify"))
df["frenzy"] = (df.text.str.contains("[fF]renzy"))
df["graft"] = (df.text.str.contains("[gG]raft"))
df["gravestorm"] = (df.text.str.contains("[gG]ravestorm"))
df["haunt"] = (df.text.str.contains("[hH]aunt"))
df["hideaway"] = (df.text.str.contains("[hH]ideaway"))
df["horsemanship"] = (df.text.str.contains("[hH]orsemanship"))
df["infect"] = (df.text.str.contains("[iI]nfect"))
df["kicker"] = (df.text.str.contains("[kK]icker"))
df["madness"] = (df.text.str.contains("[mM]adness"))
df["manifest"] = (df.text.str.contains("[mM]anifest"))
df["meld"] = (df.text.str.contains("[mM]eld"))
df["mentor"] = (df.text.str.contains("[mM]entor"))
df["miracle"] = (df.text.str.contains("[mM]iracle"))
df["modular"] = (df.text.str.contains("[mM]odular"))
df["monstrosity"] = (df.text.str.contains("[mM]onstrosity"))
df["morph"] = (df.text.str.contains("[mM]orph"))
df["multikicker"] = (df.text.str.contains("[mM]ultikicker"))
df["mutate"] = (df.text.str.contains("[mM]utate"))
df["ninjutsu"] = (df.text.str.contains("[nN]injutsu"))
df["offering"] = (df.text.str.contains("[oO]ffering"))
df["overload"] = (df.text.str.contains("[oO]verload"))
df["persist"] = (df.text.str.contains("[pP]ersist"))
df["plot"] = (df.text.str.contains("[pP]lot"))
df["poisonous"] = (df.text.str.contains("[pP]oisonous"))
df["populate"] = (df.text.str.contains("[pP]opulate"))
df["proliferate"] = (df.text.str.contains("[pP]roliferate"))
df["provoke"] = (df.text.str.contains("[pP]rovoke"))
df["prowl"] = (df.text.str.contains("[pP]rowl"))
df["radiation"] = (df.text.str.contains("[rR]adiation"))
df["rampage"] = (df.text.str.contains("[rR]ampage"))
df["rebound"] = (df.text.str.contains("[rR]ebound"))
df["recover"] = (df.text.str.contains("[rR]ecover"))
df["reinforce"] = (df.text.str.contains("[rR]einforce"))
df["renown"] = (df.text.str.contains("[rR]enown"))
df["replicate"] = (df.text.str.contains("[rR]eplicate"))
df["retrace"] = (df.text.str.contains("[rR]etrace"))
df["riot"] = (df.text.str.contains("[rR]iot"))
df["ripple"] = (df.text.str.contains("[rR]ipple"))
df["scavenge"] = (df.text.str.contains("[sS]cavenge"))
df["shadow"] = (df.text.str.contains("[sS]hadow"))
df["soulbond"] = (df.text.str.contains("[sS]oulbond"))
df["soulshift"] = (df.text.str.contains("[sS]oulshift"))
df["spectacle"] = (df.text.str.contains("[sS]pectacle"))
df["splice"] = (df.text.str.contains("[sS]plice"))
df["storm"] = (df.text.str.contains("[sS]torm"))
df["sunburst"] = (df.text.str.contains("[sS]unburst"))
df["support"] = (df.text.str.contains("[sS]upport"))
df["surveil"] = (df.text.str.contains("[sS]urveil"))
df["suspend"] = (df.text.str.contains("[sS]uspend"))
df["transfigure"] = (df.text.str.contains("[tT]ransfigure"))
df["transform"] = (df.text.str.contains("[tT]ransform"))
df["transmute"] = (df.text.str.contains("[tT]ransmute"))
df["typecycling"] = (df.text.str.contains("[tT]ypecycling"))
df["undying"] = (df.text.str.contains("[uU]ndying"))
df["unearth"] = (df.text.str.contains("[uU]nearth"))
df["unleash"] = (df.text.str.contains("[uU]nleash"))
df["vanishing"] = (df.text.str.contains("[vV]anishing"))
df["ward"] = (df.text.str.contains("[wW]ard"))
df["wither"] = (df.text.str.contains("[wW]ither"))

# card types
df["creature"] = df.type.str.contains("Creature")
df["artifact"] = df.type.str.contains("Artifact")
df["enchantment"] = df.type.str.contains("Enchantment") 
df["instant"] = df.type.str.contains("Instant") 
df["sorcery"] = df.type.str.contains("Sorcery") 
df["land"] = df.type.str.contains("Land")

# rarity
df["is_common"] = (df.rarity == "common")
df["is_uncommon"] = (df.rarity == "uncommon") 
df["is_rare"] = (df.rarity == "rare")
df["is_mythic"] = (df.rarity == "mythic")

# types
df["angel"] = df.subtypes.str.contains("Angel")
df["wall"] = df.subtypes.str.contains("Wall")
df["bird"] = df.subtypes.str.contains("Bird")
df["soldier"] = df.subtypes.str.contains("Soldier")
df["human"] = df.subtypes.str.contains("Human")
df["rebel"] = df.subtypes.str.contains("Rebel")
df["elf"] = df.subtypes.str.contains("Elf")
df["knight"] = df.subtypes.str.contains("Knight")
df["zombie"] = df.subtypes.str.contains("Zombie")
df["goblin"] = df.subtypes.str.contains("Goblin")
df["merfolk"] = df.subtypes.str.contains("Merfolk")
df["dragon"] = df.subtypes.str.contains("Dragon")
df["wizard"] = df.subtypes.str.contains("Wizard")
df["warrior"] = df.subtypes.str.contains("Warrior")
df["aura"] = df.subtypes.str.contains("Aura")
df["spirit"] = df.subtypes.str.contains("Spirit")
df["cleric"] = df.subtypes.str.contains("Cleric")
df["nomad"] = df.subtypes.str.contains("Nomad")
df["cat"] = df.subtypes.str.contains("Cat")
df["mutant"] = df.subtypes.str.contains("Mutant")
df["elephant"] = df.subtypes.str.contains("Elephant")
df["elemental"] = df.subtypes.str.contains("Elemental")
df["ogre"] = df.subtypes.str.contains("Ogre")
df["giant"] = df.subtypes.str.contains("Giant")
df["faerie"] = df.subtypes.str.contains("Faerie")
df["shapeshifter"] = df.subtypes.str.contains("Shapeshifter")
df["cephalid"] = df.subtypes.str.contains("Cephalid")
df["arcane"] = df.subtypes.str.contains("Arcane")
df["centaur"] = df.subtypes.str.contains("Centaur")
df["demon"] = df.subtypes.str.contains("Demon")
df["horror"] = df.subtypes.str.contains("Horror")
df["mercenary"] = df.subtypes.str.contains("Mercenary")
df["phyrexian"] = df.subtypes.str.contains("Phyrexian")
df["vampire"] = df.subtypes.str.contains("Vampire")
df["noble"] = df.subtypes.str.contains("Noble")
df["wraith"] = df.subtypes.str.contains("Wraith")
df["vedalken"] = df.subtypes.str.contains("Vedalken")
df["crocodile"] = df.subtypes.str.contains("Crocodile")
df["skeleton"] = df.subtypes.str.contains("Skeleton")
df["imp"] = df.subtypes.str.contains("Imp")
df["specter"] = df.subtypes.str.contains("Specter")
df["shade"] = df.subtypes.str.contains("Shade")
df["lhurgoyf"] = df.subtypes.str.contains("Lhurgoyf")
df["hydra"] = df.subtypes.str.contains("Hydra")
df["equipment"] = df.subtypes.str.contains("Equipment")
df["wurm"] = df.subtypes.str.contains("Wurm")
df["badger"] = df.subtypes.str.contains("Badger")
df["orc"] = df.subtypes.str.contains("Orc")
df["beast"] = df.subtypes.str.contains("Beast")
df["berserker"] = df.subtypes.str.contains("Berserker")
df["barbarian"] = df.subtypes.str.contains("Barbarian")
df["frog"] = df.subtypes.str.contains("Frog")
df["wolf"] = df.subtypes.str.contains("Wolf")
df["archer"] = df.subtypes.str.contains("Archer")
df["shaman"] = df.subtypes.str.contains("Shaman")
df["bear"] = df.subtypes.str.contains("Bear")
df["spider"] = df.subtypes.str.contains("Spider")
df["rat"] = df.subtypes.str.contains("Rat")
df["insect"] = df.subtypes.str.contains("Insect")
df["horse"] = df.subtypes.str.contains("Horse")
df["nightmare"] = df.subtypes.str.contains("Nightmare")
df["avatar"] = df.subtypes.str.contains("Avatar")
df["druid"] = df.subtypes.str.contains("Druid")
df["viashino"] = df.subtypes.str.contains("Viashino")
df["scout"] = df.subtypes.str.contains("Scout")
df["yeti"] = df.subtypes.str.contains("Yeti")
df["kavu"] = df.subtypes.str.contains("Kavu")
df["dryad"] = df.subtypes.str.contains("Dryad")
df["rhino"] = df.subtypes.str.contains("Rhino")
df["lizard"] = df.subtypes.str.contains("Lizard")
df["ranger"] = df.subtypes.str.contains("Ranger")
df["antelope"] = df.subtypes.str.contains("Antelope")
df["basilisk"] = df.subtypes.str.contains("Basilisk")
df["troll"] = df.subtypes.str.contains("Troll")
df["dog"] = df.subtypes.str.contains("Dog")
df["unicorn"] = df.subtypes.str.contains("Unicorn")
df["monk"] = df.subtypes.str.contains("Monk")
df["ally"] = df.subtypes.str.contains("Ally")
df["kor"] = df.subtypes.str.contains("Kor")
df["artificer"] = df.subtypes.str.contains("Artificer")
df["sphinx"] = df.subtypes.str.contains("Sphinx")
df["leviathan"] = df.subtypes.str.contains("Leviathan")
df["rogue"] = df.subtypes.str.contains("Rogue")
df["liliana"] = df.subtypes.str.contains("Liliana")
df["ooze"] = df.subtypes.str.contains("Ooze")
df["aetherborn"] = df.subtypes.str.contains("Aetherborn")
df["devil"] = df.subtypes.str.contains("Devil")
df["pirate"] = df.subtypes.str.contains("Pirate")
df["minotaur"] = df.subtypes.str.contains("Minotaur")
df["ox"] = df.subtypes.str.contains("Ox")
df["dinosaur"] = df.subtypes.str.contains("Dinosaur")
df["boar"] = df.subtypes.str.contains("Boar")
df["dwarf"] = df.subtypes.str.contains("Dwarf")
df["gargoyle"] = df.subtypes.str.contains("Gargoyle")
df["construct"] = df.subtypes.str.contains("Construct")
df["jellyfish"] = df.subtypes.str.contains("Jellyfish")
df["fish"] = df.subtypes.str.contains("Fish")
df["jace"] = df.subtypes.str.contains("Jace")
df["illusion"] = df.subtypes.str.contains("Illusion")
df["pilot"] = df.subtypes.str.contains("Pilot")
df["trap"] = df.subtypes.str.contains("Trap")
df["golem"] = df.subtypes.str.contains("Golem")
df["gremlin"] = df.subtypes.str.contains("Gremlin")
df["advisor"] = df.subtypes.str.contains("Advisor")
df["plant"] = df.subtypes.str.contains("Plant")
df["snake"] = df.subtypes.str.contains("Snake")
df["eldrazi"] = df.subtypes.str.contains("Eldrazi")
df["drone"] = df.subtypes.str.contains("Drone")
df["thopter"] = df.subtypes.str.contains("Thopter")
df["citizen"] = df.subtypes.str.contains("Citizen")
df["warlock"] = df.subtypes.str.contains("Warlock")
df["astartes"] = df.subtypes.str.contains("Astartes")
df["custodes"] = df.subtypes.str.contains("Custodes")
df["tyranid"] = df.subtypes.str.contains("Tyranid")
df["necron"] = df.subtypes.str.contains("Necron")
df["primarch"] = df.subtypes.str.contains("Primarch")
df["c'tan"] = df.subtypes.str.contains("C'tan")
df["bringer"] = df.subtypes.str.contains("Bringer")
df["griffin"] = df.subtypes.str.contains("Griffin")
df["pegasus"] = df.subtypes.str.contains("Pegasus")
df["assassin"] = df.subtypes.str.contains("Assassin")
df["thrull"] = df.subtypes.str.contains("Thrull")
df["drake"] = df.subtypes.str.contains("Drake")
df["eye"] = df.subtypes.str.contains("Eye")
df["nightstalker"] = df.subtypes.str.contains("Nightstalker")
df["minion"] = df.subtypes.str.contains("Minion")
df["goat"] = df.subtypes.str.contains("Goat")
df["cyclops"] = df.subtypes.str.contains("Cyclops")
df["ape"] = df.subtypes.str.contains("Ape")
df["ouphe"] = df.subtypes.str.contains("Ouphe")
df["hellion"] = df.subtypes.str.contains("Hellion")
df["rigger"] = df.subtypes.str.contains("Rigger")
df["turtle"] = df.subtypes.str.contains("Turtle")
df["efreet"] = df.subtypes.str.contains("Efreet")
df["treefolk"] = df.subtypes.str.contains("Treefolk")
df["nymph"] = df.subtypes.str.contains("Nymph")
df["serpent"] = df.subtypes.str.contains("Serpent")
df["octopus"] = df.subtypes.str.contains("Octopus")
df["rabbit"] = df.subtypes.str.contains("Rabbit")
df["orgg"] = df.subtypes.str.contains("Orgg")
df["hippo"] = df.subtypes.str.contains("Hippo")
df["elk"] = df.subtypes.str.contains("Elk")
df["homunculus"] = df.subtypes.str.contains("Homunculus")
df["kraken"] = df.subtypes.str.contains("Kraken")
df["carrier"] = df.subtypes.str.contains("Carrier")
df["bat"] = df.subtypes.str.contains("Bat")
df["worm"] = df.subtypes.str.contains("Worm")
df["jackal"] = df.subtypes.str.contains("Jackal")
df["spellshaper"] = df.subtypes.str.contains("Spellshaper")
df["whale"] = df.subtypes.str.contains("Whale")
df["crab"] = df.subtypes.str.contains("Crab")
df["pangolin"] = df.subtypes.str.contains("Pangolin")
df["monkey"] = df.subtypes.str.contains("Monkey")
df["bard"] = df.subtypes.str.contains("Bard")
df["phoenix"] = df.subtypes.str.contains("Phoenix")
df["sheep"] = df.subtypes.str.contains("Sheep")
df["gnome"] = df.subtypes.str.contains("Gnome")
df["beholder"] = df.subtypes.str.contains("Beholder")
df["tiefling"] = df.subtypes.str.contains("Tiefling")
df["curse"] = df.subtypes.str.contains("Curse")
df["flagbearer"] = df.subtypes.str.contains("Flagbearer")
df["volver"] = df.subtypes.str.contains("Volver")
df["metathran"] = df.subtypes.str.contains("Metathran")
df["tamiyo"] = df.subtypes.str.contains("Tamiyo")
df["tibalt"] = df.subtypes.str.contains("Tibalt")
df["will"] = df.subtypes.str.contains("Will")
df["azra"] = df.subtypes.str.contains("Azra")
df["rowan"] = df.subtypes.str.contains("Rowan")
df["fox"] = df.subtypes.str.contains("Fox")
df["djinn"] = df.subtypes.str.contains("Djinn")
df["mordenkainen"] = df.subtypes.str.contains("Mordenkainen")
df["god"] = df.subtypes.str.contains("God")
df["gideon"] = df.subtypes.str.contains("Gideon")
df["camel"] = df.subtypes.str.contains("Camel")
df["cartouche"] = df.subtypes.str.contains("Cartouche")
df["naga"] = df.subtypes.str.contains("Naga")
df["scorpion"] = df.subtypes.str.contains("Scorpion")
df["manticore"] = df.subtypes.str.contains("Manticore")
df["hyena"] = df.subtypes.str.contains("Hyena")
df["chandra"] = df.subtypes.str.contains("Chandra")
df["chimera"] = df.subtypes.str.contains("Chimera")
df["elspeth"] = df.subtypes.str.contains("Elspeth")
df["sliver"] = df.subtypes.str.contains("Sliver")
df["squid"] = df.subtypes.str.contains("Squid")
df["hippogriff"] = df.subtypes.str.contains("Hippogriff")
df["mongoose"] = df.subtypes.str.contains("Mongoose")
df["kithkin"] = df.subtypes.str.contains("Kithkin")
df["archon"] = df.subtypes.str.contains("Archon")
df["treasure"] = df.subtypes.str.contains("Treasure")
df["satyr"] = df.subtypes.str.contains("Satyr")
df["nixilis"] = df.subtypes.str.contains("Nixilis")
df["fungus"] = df.subtypes.str.contains("Fungus")
df["mouse"] = df.subtypes.str.contains("Mouse")
df["peasant"] = df.subtypes.str.contains("Peasant")
df["background"] = df.subtypes.str.contains("Background")
df["gith"] = df.subtypes.str.contains("Gith")
df["halfling"] = df.subtypes.str.contains("Halfling")
df["kobold"] = df.subtypes.str.contains("Kobold")
df["elder"] = df.subtypes.str.contains("Elder")
df["egg"] = df.subtypes.str.contains("Egg")
df["siren"] = df.subtypes.str.contains("Siren")
df["harpy"] = df.subtypes.str.contains("Harpy")
df["gorgon"] = df.subtypes.str.contains("Gorgon")
df["nissa"] = df.subtypes.str.contains("Nissa")
df["wolverine"] = df.subtypes.str.contains("Wolverine")
df["mount"] = df.subtypes.str.contains("Mount")
df["ninja"] = df.subtypes.str.contains("Ninja")
df["robot"] = df.subtypes.str.contains("Robot")
df["synth"] = df.subtypes.str.contains("Synth")
df["detective"] = df.subtypes.str.contains("Detective")
df["vehicle"] = df.subtypes.str.contains("Vehicle")
df["slug"] = df.subtypes.str.contains("Slug")
df["alien"] = df.subtypes.str.contains("Alien")
df["guest"] = df.subtypes.str.contains("Guest")
df["performer"] = df.subtypes.str.contains("Performer")
df["clown"] = df.subtypes.str.contains("Clown")
df["employee"] = df.subtypes.str.contains("Employee")
df["homarid"] = df.subtypes.str.contains("Homarid")
df["praetor"] = df.subtypes.str.contains("Praetor")
df["moonfolk"] = df.subtypes.str.contains("Moonfolk")
df["shrine"] = df.subtypes.str.contains("Shrine")
df["leech"] = df.subtypes.str.contains("Leech")
df["soltari"] = df.subtypes.str.contains("Soltari")
df["incarnation"] = df.subtypes.str.contains("Incarnation")
df["salamander"] = df.subtypes.str.contains("Salamander")
df["scientist"] = df.subtypes.str.contains("Scientist")
df["samurai"] = df.subtypes.str.contains("Samurai")
df["squirrel"] = df.subtypes.str.contains("Squirrel")
df["lesson"] = df.subtypes.str.contains("Lesson")
df["mystic"] = df.type.str.contains("Mystic")





df[["power", "toughness"]] = df[["power", "toughness"]].replace("\\d*\\+*\\*|\\?|none|∞", 0, regex = True)
df = df.astype({"power": "float", "toughness": "float"})

df["is_white"] = (df.color_identity.str.contains('W'))
df["is_blue"] = (df.color_identity.str.contains('U'))
df["is_black"] = (df.color_identity.str.contains('B'))
df["is_red"] = (df.color_identity.str.contains('R'))
df["is_green"] = (df.color_identity.str.contains('G'))

output_file = 'all_mtg_cards_feature_abilities.csv'
df.to_csv(output_file, index=False)


