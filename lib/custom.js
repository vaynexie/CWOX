// this file assumes a variable nodes is used to hold the nodes of the jstree
// and a <div> with id 'jstree' as a placeholder of the tree.

//for translate classname to image index
var name_to_index = {"bee": "309", "fly": "308", "packet": "692", "envelope": "549", "obelisk": "682", "pedestal": "708", "slide_rule": "798", "dishrag": "533", "strawberry": "949", "pomegranate": "957", "miniature_pinscher": "237", "kelpie": "227", "restaurant": "762", "bakery": "415", "indigo_bunting": "014", "peacock": "084", "trombone": "875", "cornet": "513", "Kerry_blue_terrier": "183", "Scotch_terrier": "199", "cauliflower": "938", "broccoli": "937", "hammer": "587", "screwdriver": "784", "sea_urchin": "328", "lionfish": "396", "broom": "462", "swab": "840", "earthstar": "995", "hen-of-the-woods": "996", "sandbar": "977", "seashore": "978", "groenendael": "224", "schipperke": "223", "borzoi": "169", "Saluki": "176", "sulphur_butterfly": "325", "cabbage_butterfly": "324", "pier": "718", "suspension_bridge": "839", "plastic_bag": "728", "umbrella": "879", "robin": "015", "quail": "085", "wire-haired_fox_terrier": "188", "Bedlington_terrier": "181", "carton": "478", "crate": "519", "monarch": "323", "admiral": "321", "Gila_monster": "045", "banded_gecko": "038", "Doberman": "236", "Rottweiler": "234", "triumphal_arch": "873", "palace": "698", "sliding_door": "799", "shoji": "789", "table_lamp": "846", "lampshade": "619", "drumstick": "542", "drum": "541", "water_ouzel": "020", "American_coot": "137", "cuirass": "524", "breastplate": "461", "corn": "987", "ear": "998", "Greater_Swiss_Mountain_dog": "238", "Bernese_mountain_dog": "239", "Polaroid_camera": "732", "reflex_camera": "759", "electric_fan": "545", "spotlight": "818", "Persian_cat": "283", "Angora": "332", "water_buffalo": "346", "warthog": "343", "switch": "844", "combination_lock": "507", "grocery_store": "582", "confectionery": "509", "mortar": "666", "Crock_Pot": "521", "dough": "961", "picket_fence": "716", "worm_fence": "912", "Great_Pyrenees": "257", "kuvasz": "222", "beer_bottle": "440", "pop_bottle": "737", "sunscreen": "838", "lotion": "631", "cowboy_hat": "515", "sombrero": "808", "redshank": "141", "ruddy_turnstone": "139", "Lhasa": "204", "Shih-Tzu": "155", "fire_screen": "556", "guillotine": "583", "safety_pin": "772", "hair_slide": "584", "cheetah": "293", "snow_leopard": "289", "shower_cap": "793", "neck_brace": "678", "harvester": "595", "thresher": "856", "tractor": "866", "police_van": "734", "ambulance": "407", "Arctic_fox": "279", "white_wolf": "270", "goldfish": "001", "axolotl": "029", "Italian_greyhound": "171", "whippet": "172", "pole": "733", "flagpole": "557", "king_penguin": "145", "ice_bear": "296", "grey_whale": "147", "killer_whale": "148", "radio_telescope": "755", "solar_dish": "807", "printer": "742", "photocopier": "713", "piggy_bank": "719", "ocarina": "684", "abaya": "399", "cloak": "501", "speedboat": "814", "amphibian": "408", "paddlewheel": "694", "steel_arch_bridge": "821", "sloth_bear": "297", "American_black_bear": "295", "coil": "506", "knot": "616", "Arabian_camel": "354", "llama": "355", "colobus": "375", "giant_panda": "388", "indri": "384", "Madagascar_cat": "383", "rubber_eraser": "767", "ballpoint": "418", "trench_coat": "869", "fur_coat": "568", "clog": "502", "sandal": "774", "rain_barrel": "756", "barrel": "427", "waffle_iron": "891", "toaster": "859", "tripod": "872", "binoculars": "447", "Cardigan": "264", "Pembroke": "263", "sewing_machine": "786", "joystick": "613", "shovel": "792", "hatchet": "596", "maillot-c639": "639", "maillot-c638": "638", "miniskirt": "655", "jean": "608", "wooden_spoon": "910", "spatula": "813", "stretcher": "830", "military_uniform": "652", "CD_player": "485", "radio": "754", "cellular_telephone": "487", "remote_control": "761", "screw": "783", "nail": "677", "bassinet": "431", "cradle": "516", "church": "497", "monastery": "663", "maraca": "641", "ping-pong_ball": "722", "pelican": "144", "goose": "099", "agaric": "992", "hip": "989", "tow_truck": "864", "trailer_truck": "867", "leopard": "288", "jaguar": "290", "street_sign": "919", "traffic_light": "920", "panpipe": "699", "harp": "594", "wolf_spider": "077", "tarantula": "076", "face_powder": "551", "lipstick": "629", "pretzel": "932", "cheeseburger": "933", "soft-coated_wheaten_terrier": "202", "Irish_terrier": "184", "ruffed_grouse": "082", "partridge": "086", "mixing_bowl": "659", "soup_bowl": "809", "golden_retriever": "207", "Irish_setter": "213", "mobile_home": "660", "freight_car": "565", "cicada": "316", "rhinoceros_beetle": "306", "mailbag": "636", "backpack": "414", "ladybug": "301", "leaf_beetle": "304", "dome": "538", "mosque": "668", "grey_fox": "280", "fox_squirrel": "335", "long-horned_beetle": "303", "weevil": "307", "Siamese_cat": "284", "black-footed_ferret": "359", "Yorkshire_terrier": "187", "silky_terrier": "201", "thunder_snake": "052", "ringneck_snake": "053", "water_snake": "058", "sea_snake": "065", "horned_viper": "066", "sidewinder": "068", "snowmobile": "802", "bobsled": "450", "perfume": "711", "water_bottle": "898", "daisy": "985", "rapeseed": "984", "Mexican_hairless": "268", "African_grey": "087", "great_grey_owl": "024", "magnetic_compass": "635", "barometer": "426", "fireboat": "554", "lifeboat": "625", "bell_cote": "442", "vault": "884", "ptarmigan": "081", "bustard": "138", "king_snake": "056", "diamondback": "067", "marmot": "336", "cliff_dwelling": "500", "Brabancon_griffon": "262", "affenpinscher": "252", "spider_web": "815", "black_and_gold_garden_spider": "072", "steam_locomotive": "820", "electric_locomotive": "547", "Tibetan_terrier": "200", "briard": "226", "hand-held_computer": "590", "oscilloscope": "688", "plow": "730", "cannon": "471", "plate": "923", "mashed_potato": "935", "orange": "950", "lemon": "951", "terrapin": "036", "mud_turtle": "035", "horizontal_bar": "602", "parallel_bars": "702", "Pekinese": "154", "Pomeranian": "259", "beaker": "438", "measuring_cup": "647", "otterhound": "175", "Dandie_Dinmont": "194", "shoe_shop": "788", "toyshop": "865", "plate_rack": "729", "medicine_chest": "648", "standard_schnauzer": "198", "miniature_schnauzer": "196", "crossword_puzzle": "918", "quill": "749", "red-breasted_merganser": "098", "drake": "097", "stupa": "832", "castle": "483", "wine_bottle": "907", "red_wine": "966", "bicycle-built-for-two": "444", "mountain_bike": "671", "comic_book": "917", "book_jacket": "921", "Egyptian_cat": "285", "lynx": "287", "wombat": "106", "wallaby": "104", "laptop": "620", "notebook": "681", "flamingo": "130", "sorrel": "339", "jack-o'-lantern": "607", "whiskey_jug": "901", "velvet": "885", "cardigan": "474", "Saint_Bernard": "247", "Blenheim_spaniel": "156", "teddy": "850", "Christmas_stocking": "496", "beaver": "337", "platypus": "103", "frilled_lizard": "043", "agama": "042", "echidna": "102", "porcupine": "334", "oil_filter": "686", "disk_brake": "535", "sulphur-crested_cockatoo": "089", "spoonbill": "129", "wig": "903", "feather_boa": "552", "crash_helmet": "518", "knee_pad": "615", "football_helmet": "560", "red-backed_sandpiper": "140", "dowitcher": "142", "shower_curtain": "794", "mosquito_net": "669", "bald_eagle": "022", "vulture": "023", "kite": "021", "bolo_tie": "451", "Windsor_tie": "906", "flat-coated_retriever": "205", "Newfoundland": "256", "paddle": "693", "canoe": "472", "alp": "970", "valley": "979", "unicycle": "880", "tricycle": "870", "German_short-haired_pointer": "210", "bluetick": "164", "butternut_squash": "942", "spaghetti_squash": "940", "Staffordshire_bullterrier": "179", "American_Staffordshire_terrier": "180", "barbershop": "424", "barber_chair": "423", "cleaver": "499", "power_drill": "740", "tank": "847", "half_track": "586", "chain_saw": "491", "lawn_mower": "621", "ice_lolly": "929", "Band_Aid": "419", "cocker_spaniel": "219", "Sussex_spaniel": "220", "grand_piano": "579", "upright": "881", "swimming_trunks": "842", "bikini": "445", "harvestman": "070", "spiny_lobster": "123", "sea_cucumber": "329", "starfish": "327", "pitcher": "725", "water_jug": "899", "pencil_sharpener": "710", "lighter": "626", "organ": "687", "theater_curtain": "854", "koala": "105", "three-toed_sloth": "364", "Dungeness_crab": "118", "king_crab": "121", "cab": "468", "limousine": "627", "flatworm": "110", "sea_slug": "115", "acoustic_guitar": "402", "electric_guitar": "546", "burrito": "965", "hotdog": "934", "car_mirror": "475", "hourglass": "604", "hognose_snake": "054", "night_snake": "060", "moving_van": "675", "garbage_truck": "569", "brain_coral": "109", "coral_reef": "973", "washer": "897", "toilet_seat": "861", "stinkhorn": "994", "gyromitra": "993", "coral_fungus": "991", "coyote": "272", "timber_wolf": "269", "dugong": "149", "hammerhead": "004", "gas_pump": "571", "turnstile": "877", "buckle": "464", "pick": "714", "dogsled": "537", "ski": "795", "oxcart": "690", "ox": "345", "vizsla": "211", "bloodhound": "163", "balloon": "417", "parachute": "701", "vestment": "887", "jigsaw_puzzle": "611", "megalith": "649", "stone_wall": "825", "tray": "868", "scale": "778", "African_hunting_dog": "275", "hyena": "276", "Siberian_husky": "250", "Eskimo_dog": "248", "ladle": "618", "strainer": "828", "seat_belt": "785", "sleeping_bag": "797", "school_bus": "779", "fire_engine": "555", "jeep": "609", "snowplow": "803", "desk": "526", "desktop_computer": "527", "zucchini": "939", "cucumber": "943", "jacamar": "095", "bee_eater": "092", "sunglass": "836", "sunglasses": "837", "racer": "751", "sports_car": "817", "pool_table": "736", "potter's_wheel": "739", "weasel": "356", "polecat": "358", "wreck": "913", "container_ship": "510", "stage": "819", "microphone": "650", "patio": "706", "greenhouse": "580", "caldron": "469", "Dutch_oven": "544", "house_finch": "012", "coucal": "091", "bubble": "971", "fountain": "562", "tabby": "281", "tiger_cat": "282", "stethoscope": "823", "lab_coat": "617", "capuchin": "378", "squirrel_monkey": "382", "apiary": "410", "honeycomb": "599", "sturgeon": "394", "coho": "391", "espresso": "967", "eggnog": "969", "custard_apple": "956", "jackfruit": "955", "meerkat": "299", "mongoose": "298", "puck": "746", "running_shoe": "770", "EntleBucher": "241", "Appenzeller": "240", "tennis_ball": "852", "racket": "752", "Australian_terrier": "193", "Shetland_sheepdog": "230", "centipede": "079", "scorpion": "071", "hand_blower": "589", "hair_spray": "585", "black-and-tan_coonhound": "165", "Gordon_setter": "214", "trimaran": "871", "catamaran": "484", "gorilla": "366", "orangutan": "365", "mushroom": "947", "bolete": "997", "diaper": "529", "nipple": "680", "cowboy_boot": "514", "Loafer": "630", "dam": "525", "viaduct": "888", "golf_ball": "574", "croquet_ball": "522", "mountain_tent": "672", "yurt": "915", "chime": "494", "gong": "577", "tiger_beetle": "300", "ground_beetle": "302", "dung_beetle": "305", "cockroach": "314", "ant": "310", "space_heater": "811", "radiator": "753", "frying_pan": "567", "wok": "909", "typewriter_keyboard": "878", "space_bar": "810", "paper_towel": "700", "toilet_tissue": "999", "dishwasher": "534", "refrigerator": "760", "moped": "665", "motor_scooter": "670", "stole": "824", "wool": "911", "poncho": "735", "forklift": "561", "lumbermill": "634", "little_blue_heron": "131", "European_gallinule": "136", "meat_loaf": "962", "potpie": "964", "skunk": "361", "badger": "362", "papillon": "157", "Japanese_spaniel": "152", "menu": "922", "web_site": "916", "curly-coated_retriever": "206", "black_swan": "100", "balance_beam": "416", "bathing_cap": "433", "bull_mastiff": "243", "boxer": "242", "leatherback_turtle": "034", "loggerhead": "033", "schooner": "780", "pirate": "724", "rock_crab": "119", "fiddler_crab": "120", "goblet": "572", "beer_glass": "441", "toucan": "096", "hornbill": "093", "African_chameleon": "047", "vine_snake": "059", "plunger": "731", "crutch": "523", "pickelhaube": "715", "bulletproof_vest": "465", "baseball": "429", "ballplayer": "981", "Weimaraner": "178", "Great_Dane": "246", "lorikeet": "090", "macaw": "088", "carpenter's_kit": "477", "fountain_pen": "563", "proboscis_monkey": "376", "lesser_panda": "387", "stopwatch": "826", "digital_watch": "531", "tiger_shark": "003", "great_white_shark": "002", "golfcart": "575", "Model_T": "661", "can_opener": "473", "corkscrew": "512", "china_cabinet": "495", "wardrobe": "894", "promontory": "976", "cliff": "972", "alligator_lizard": "044", "whiptail": "041", "overskirt": "689", "hoopskirt": "601", "rocking_chair": "765", "folding_chair": "559", "altar": "406", "throne": "857", "matchstick": "644", "abacus": "398", "crayfish": "124", "American_lobster": "122", "Old_English_sheepdog": "229", "komondor": "228", "leafhopper": "317", "lacewing": "318", "common_newt": "026", "eft": "027", "hay": "958", "maze": "646", "barn_spider": "073", "garden_spider": "074", "carousel": "476", "pinwheel": "723", "cassette_player": "482", "tape_player": "848", "crane-c134": "134", "American_egret": "132", "minivan": "656", "beach_wagon": "436", "ibex": "350", "hartebeest": "351", "patas": "371", "guenon": "370", "liner": "628", "dock": "536", "chow": "260", "keeshond": "261", "passenger_car": "705", "trolleybus": "874", "streetcar": "829", "banana": "954", "pineapple": "953", "Petri_dish": "712", "nematode": "111", "library": "624", "bookshop": "454", "tub": "876", "bathtub": "435", "rifle": "764", "assault_rifle": "413", "English_springer": "217", "English_setter": "212", "totem_pole": "863", "scoreboard": "781", "chickadee": "019", "goldfinch": "011", "Irish_water_spaniel": "221", "standard_poodle": "267", "red_wolf": "271", "dhole": "274", "vase": "883", "candle": "470", "projector": "745", "loudspeaker": "632", "cairn": "192", "Border_terrier": "182", "prayer_rug": "741", "doormat": "539", "wild_boar": "342", "hog": "341", "gown": "578", "groom": "982", "marimba": "642", "steel_drum": "822", "howler_monkey": "379", "spider_monkey": "381", "hermit_crab": "125", "conch": "112", "tench": "000", "head_cabbage": "936", "violin": "889", "cello": "486", "cash_machine": "480", "pay-phone": "707", "lens_cap": "622", "loupe": "633", "vending_machine": "886", "slot": "800", "rock_python": "062", "boa_constrictor": "061", "accordion": "401", "banjo": "420", "bookcase": "453", "file": "553", "oxygen_mask": "691", "gasmask": "570", "mortarboard": "667", "academic_gown": "400", "malamute": "249", "Norwegian_elkhound": "174", "sundial": "835", "brass": "458", "fig": "952", "Granny_Smith": "948", "cricket": "312", "grasshopper": "311", "Tibetan_mastiff": "244", "Leonberg": "255", "anemone_fish": "393", "sea_anemone": "108", "microwave": "651", "stove": "827", "aircraft_carrier": "403", "submarine": "833", "muzzle": "676", "pug": "254", "bullet_train": "466", "odometer": "685", "bow_tie": "457", "suit": "834", "swing": "843", "maypole": "645", "cardoon": "946", "artichoke": "944", "tile_roof": "858", "thatch": "853", "crane-c517": "517", "drilling_platform": "540", "rotisserie": "766", "butcher_shop": "467", "dragonfly": "319", "damselfly": "320", "isopod": "126", "chiton": "116", "chainlink_fence": "489", "shopping_cart": "791", "necklace": "679", "spindle": "816", "dingo": "273", "cougar": "286", "basenji": "253", "Ibizan_hound": "173", "warplane": "895", "wing": "908", "siamang": "369", "chimpanzee": "367", "snorkel": "801", "scuba_diver": "983", "bath_towel": "434", "handkerchief": "591", "beacon": "437", "water_tower": "900", "binder": "446", "cassette": "481", "boathouse": "449", "barn": "425", "American_alligator": "050", "African_crocodile": "049", "green_snake": "055", "green_mamba": "064", "convertible": "511", "grille": "581", "pillow": "721", "quilt": "750", "German_shepherd": "235", "malinois": "225", "dumbbell": "543", "barbell": "422", "hot_pot": "926", "consomme": "925", "espresso_maker": "550", "cocktail_shaker": "503", "bulbul": "016", "hummingbird": "094", "basketball": "430", "volleyball": "890", "Rhodesian_ridgeback": "159", "redbone": "168", "padlock": "695", "birdhouse": "448", "chocolate_sauce": "960", "ice_cream": "928", "manhole_cover": "640", "trilobite": "069", "computer_keyboard": "508", "mouse": "673", "pencil_box": "709", "rule": "769", "Chesapeake_Bay_retriever": "209", "Labrador_retriever": "208", "projectile": "744", "missile": "657", "harmonica": "593", "whistle": "902", "wallet": "893", "purse": "748", "paintbrush": "696", "syringe": "845", "dalmatian": "251", "Samoyed": "258", "pot": "738", "milk_can": "653", "park_bench": "703", "barrow": "428", "baboon": "372", "macaque": "373", "bell_pepper": "945", "acorn_squash": "941", "iPod": "605", "digital_clock": "530", "bighorn": "349", "ram": "348", "tree_frog": "031", "rock_beauty": "392", "yellow_lady's_slipper": "986", "washbasin": "896", "soap_dispenser": "804", "television": "851", "home_theater": "598", "vacuum": "882", "iron": "606", "ringlet": "322", "lycaenid": "326", "reel": "758", "bow": "456", "white_stork": "127", "black_stork": "128", "volcano": "980", "geyser": "974", "gibbon": "368", "langur": "374", "tusker": "101", "Indian_elephant": "385", "bib": "443", "bonnet": "452", "Airedale": "191", "Lakeland_terrier": "189", "triceratops": "051", "African_elephant": "386", "scabbard": "777", "letter_opener": "623", "bucket": "463", "ashcan": "412", "guinea_pig": "338", "hamster": "333", "planetarium": "727", "airship": "405", "albatross": "146", "oystercatcher": "143", "mousetrap": "674", "plane": "726", "monitor": "664", "screen": "782", "safe": "771", "chest": "492", "cock": "007", "hen": "008", "space_shuttle": "812", "airliner": "404", "dial_telephone": "528", "French_horn": "566", "dining_table": "532", "crib": "520", "pill_bottle": "720", "bottlecap": "455", "stingray": "006", "electric_ray": "005", "chain_mail": "490", "shield": "787", "toy_poodle": "265", "miniature_poodle": "266", "bittern": "133", "limpkin": "135", "giant_schnauzer": "197", "Bouvier_des_Flandres": "233", "breakwater": "460", "lakeside": "975", "shopping_basket": "790", "hamper": "588", "horse_cart": "603", "jinrikisha": "612", "impala": "352", "gazelle": "353", "collie": "231", "Border_collie": "232", "bullfrog": "030", "tailed_frog": "032", "tobacco_shop": "860", "cinema": "498", "Irish_wolfhound": "170", "Scottish_deerhound": "177", "Brittany_spaniel": "215", "Welsh_springer_spaniel": "218", "gondola": "576", "yawl": "914", "go-kart": "573", "rugby_ball": "768", "soccer_ball": "805", "wood_rabbit": "330", "hare": "331", "bagel": "931", "French_loaf": "930", "pickup": "717", "car_wheel": "479", "acorn": "988", "buckeye": "990", "prairie_chicken": "083", "black_grouse": "080", "common_iguana": "039", "Komodo_dragon": "048", "brassiere": "459", "punching_bag": "747", "puffer": "397", "eel": "390", "brambling": "010", "junco": "013", "parking_meter": "704", "mailbox": "637", "bassoon": "432", "sax": "776", "lion": "291", "tiger": "292", "spotted_salamander": "028", "European_fire_salamander": "025", "jay": "017", "magpie": "018", "jellyfish": "107", "chambered_nautilus": "117", "Norfolk_terrier": "185", "Norwich_terrier": "186", "zebra": "340", "ostrich": "009", "beagle": "162", "Walker_hound": "166", "Afghan_hound": "160", "Sealyham_terrier": "190", "clumber": "216", "prison": "743", "bannister": "421", "barracouta": "389", "gar": "395", "green_lizard": "046", "American_chameleon": "040", "English_foxhound": "167", "basset": "161", "mask": "643", "ski_mask": "796", "hook": "600", "chain": "488", "garter_snake": "057", "Indian_cobra": "063", "Chihuahua": "151", "toy_terrier": "158", "torch": "862", "bearskin": "439", "mitten": "658", "sock": "806", "French_bulldog": "245", "Boston_bull": "195", "bison": "347", "brown_bear": "294", "window_shade": "905", "window_screen": "904", "saltshaker": "773", "thimble": "855", "apron": "411", "sarong": "775", "slug": "114", "snail": "113", "jersey": "610", "sweatshirt": "841", "four-poster": "564", "studio_couch": "831", "titi": "380", "marmoset": "377", "West_Highland_white_terrier": "203", "Maltese_dog": "153", "oboe": "683", "flute": "558", "tick": "078", "black_widow": "075", "pizza": "963", "trifle": "927", "coffeepot": "505", "teapot": "849", "revolver": "763", "holster": "597", "guacamole": "924", "carbonara": "959", "red_fox": "277", "kit_fox": "278", "kimono": "614", "pajama": "697", "mantis": "315", "walking_stick": "313", "recreational_vehicle": "757", "minibus": "654", "wall_clock": "892", "analog_clock": "409", "armadillo": "363", "box_turtle": "037", "hard_disc": "592", "modem": "662", "mink": "357", "otter": "360", "entertainment_center": "548", "chiffonier": "493", "sea_lion": "150", "hippopotamus": "344", "cup": "968", "coffee_mug": "504"}

function getInputValue(id, defaultValue) {
	var value = $(id).val();
	if (value.length <= 0)
		value = defaultValue;

	return parseInt(value);
}

function findLevel(node) {
	var l = node.data.level - 1;
	if (typeof levels[l] == 'undefined')
		levels[l] = [];

	levels[l].push(node)

	$.each(node.children, function(i, v) {
		findLevel(v);
	})
}

// find the node levels in the tree
var levels = [];
$.each(nodes, function(i, v) {
	findLevel(v)
});

// find the min and max year in the documents
var showTopicDocuments = typeof documents != "undefined"
		&& typeof topicMap != "undefined"

if (showTopicDocuments) {
	var minYear = 1000000;
	var maxYear = 0;
	$.each(documents, function(i, d) {
		if (d.year > maxYear)
			maxYear = d.year;
		if (d.year < minYear)
			minYear = d.year;
	})
}

function generateDocumentPage(content){
	window.open('about:blank', '_blank').document.body.innerText += content;
}

function generateTopicDocumentTable(topic, max) {
	var topicDocuments = topicMap[topic];

	var rows = [];
	
	if(typeof(fieldnames) !== 'undefined'){
		var title = fieldnames.indexOf('title');
		var url = fieldnames.indexOf('url');
		for (var i = 0; i < topicDocuments.length && i < max; i++) {
			var d = topicDocuments[i];
			var doc = documents[d[0]]; //documents is an array of document name
			
			var columns = [];
			if(url != -1 && doc[url].length>0) columns.push("<a href=\"" + doc[url] + "\">" + doc[title] + "</a>");
			else columns.push(doc[title]);
			for(var j=0; j<doc.length; j++){
				if(j==url || j==title)
					continue;
				columns.push(doc[j]);
			}
			columns.push(d[1].toFixed(2));
			rows.push("<tr><td>" + columns.join("</td><td>") + "</td></tr>");
		}
		var otherFields = ["Document"];
		for(var j=0; j<fieldnames.length; j++){
			if(j==url || j==title)
				continue;
			otherFields.push(fieldnames[j]);
		}
		otherFields.push("Prob");
		var table = $("<table class=\"tablesorter\"><thead><tr><th>"+otherFields.join("</th><th>")+"</th></tr></thead></table>")
				.append("<tbody/>").append(rows.join(""));
	}else{
		for (var i = 0; i < topicDocuments.length && i < max; i++) {
			var d = topicDocuments[i];
			var doc = documents[d[0]]; //documents is an array of document name

			if(Array.isArray(doc)){//doc can either be in the form of Array("someTitle", "someUrl", "somethingElse1", ...) or String("titleOnly")
				rows.push("<tr><td><a href=\"" + doc[1] + "\">" + doc[0] + "</a></td><td>" + d[1].toFixed(2)+ "</td></tr>");
			}else{
				if(doc.length > 65) rows.push("<tr><td><a href=\"#\" onclick=\"generateDocumentPage('"+doc+"')\">" + doc + "...</a></td><td>" + d[1].toFixed(2)+ "</td></tr>");
				else rows.push("<tr><td>" +doc + "</td><td>" + d[1].toFixed(2)+ "</td></tr>");
			}
		}
		var table = $("<table class=\"tablesorter\"><thead><tr><th>Document</th><th>Prob</th></tr></thead></table>")
			.append("<tbody/>").append(rows.join(""));
	}

	

	table.tablesorter({
		theme : "bootstrap",
		widthFixed : true,
		headerTemplate : '{content} {icon}',
		widgets : [ "uitheme", "zebra" ],
		widgetOptions : {
			zebra : [ "even", "odd" ],
		}
	});

	return table;
}

function generateCountTable(topic) {
	var topicDocuments = topicMap[topic];
	var counts = {};
	for (var year = minYear; year <= maxYear; year++) {
		counts[year] = 0;
	}

	$.each(topicDocuments, function(i, d) {
		var doc = documents[d[0]];
		counts[doc.year] = counts[doc.year] + 1;
	})

	var headRow = $("<tr/>");
	var bodyRow = $("<tr/>");
	for (var year = minYear; year <= maxYear; year++) {
		headRow.append("<th>" + year + "</th>");
		bodyRow.append("<td>" + counts[year] + "</td>");
	}
	var table = $("<table class=\"table table-bordered table-condensed\"/>")
			.append("<thead/>").append("<tbody/>");
	table.children("thead").append(headRow);
	table.children("tbody").append(bodyRow);

	return table;
}

function constructTree(n) {
	$("#jstree").on("changed.jstree", function(e, data) {
		// show a pop-up when a node has been selected
		if (data.action == "select_node") {
			$("#topic-modal-title").html(
				data.node.text + " (" + data.node.id + ")")

			$("#topic-modal-body").html("")
	
			if (showTopicDocuments) {
				var topicDocumentsCount = topicMap[data.node.id].length;
				var max = 50000;
	
				if(topicDocumentsCount > max) $("#topic-modal-body").append("<h5>Document details (showing only the top " + max +"):</h5>");
				else $("#topic-modal-body").append("<h5>Document details (" + topicDocumentsCount + " documents):</h5>");
				$("#topic-modal-body").append(generateTopicDocumentTable(
						data.node.id, max));
			} else {
				$("#topic-modal-body").append("<p>Document information is not available.</p>")
			}

			$("#topic-modal").modal()
		}
	}).on("open_node.jstree", function(e, data) {
		var target = $("li.jstree-leaf").filter(function() {return data.node.children_d.includes($(this).attr("id"))});
		load_image(target);
	}).on("ready.jstree", function(e, data) {
		var alltext = $("li.jstree-leaf").children(".jstree-anchor").text().split(" ");
		if (alltext.every(d => /c\d{3}$/.test(d))) {
			text_to_index = d => d.replaceAll("c", "").split(" ");
		}
		else if (alltext.every(d => /c\d{3}-/.test(d))) {
			text_to_index = d => d.split(" ").map(x => x.split("-", 1)[0].replace("c", ""));
		}
		else {
			text_to_index = d => d.split(" ").map(x => name_to_index[x]);
		}
		$.fn.isInViewport = function () {
			let elementTop = $(this).offset().top;
			let elementBottom = elementTop + $(this).outerHeight();
			let viewportTop = $(window).scrollTop();
			let viewportBottom = viewportTop + $(window).height();
			return elementBottom > viewportTop && elementTop < viewportBottom;
		};

		index_to_name = Object.fromEntries(Object.entries(name_to_index).map(a => a.reverse()));
		var target = $("li.jstree-leaf").filter(function() {return $(this).isInViewport()});
		load_image(target);
		$(window).on("resize scroll", function () {
			var target = $("li.jstree-leaf").filter(function() {return $(this).isInViewport()});
			load_image(target);
		});

	}).jstree({
				"core" : {
					"data" : n,
					"themes" : {
						"icons" : false
					}
				},
				"search" : {
					"case_insensitive" : true,
					"show_only_matches" : true,
					"show_only_matches_children" : true
				},
				"plugins" : [ "search" ]
			});
}

function load_image(target) {
	target = target.filter(function() {return $(this).attr("data-image-loaded") != "true"});
	target.attr("data-image-loaded", true);
	target.append(function () {
		var image_list = text_to_index($(this).children(".jstree-anchor").text());
		var append = image_list.map(d => `<div style="display:inline-block;"><div><span>${index_to_name[d]}</span></div><div><img src="class_image/${parseInt(d)}.png" height="100px"></img></div></div>`).join(" ");
		return `<div style="display:inline-block;">${append}</div>`;
	})
	target.children(".jstree-anchor").remove();
}




// show the node within the specified range of levels.
// the nodes above the topmost level are discarded, while
// the nodes below the bottommost level are closed.
function showLevels(top, bottom) {
	var current = $('#jstree').jstree(true);
	if (typeof current != 'undefined' && current)
		current.destroy();

	for (var i = top; i > bottom; i--) {
		$.each(levels[i - 1], function(i, v) {
			if(typeof(v.state) === "undefined")
				v.state = {opened: true}
			else
				v.state.opened = true;
		})
	}

	for (var i = bottom; i > 0; i--) {
		$.each(levels[i - 1], function(i, v) {
			if(typeof(v.state) === "undefined")
				v.state = {opened: false}
			else
				v.state.opened = false;
		})
	}

	constructTree(levels[top - 1]);
}

function showAlert(message) {
	$("#alert-modal-message").html(message)
	$("#alert-modal").modal()
}

$(function() {
	topmost = levels.length
	//bottommost = Math.max(1, levels.length - 1)
	bottommost = 1

	// set the default values of the levels
	$("#top-input").val(topmost)
	$("#bottom-input").val(bottommost)

	$('[data-toggle="tooltip"]').tooltip()

	showLevels(topmost, bottommost);

	$('#level-button').click(function() {
		var top = getInputValue('#top-input', 1000000)
		var bottom = getInputValue('#bottom-input', 1)

		if (top > levels.length) {
			showAlert("The topmost level (left) cannot be larger than "
					+ levels.length + ".")
			$("#top-input").val(topmost)
		} else if (bottom < 1) {
			showAlert("The bottommost level (right) cannot be smaller than 1.")
			$("#bottom-input").val(1)
		} else if (top < bottom) {
			showAlert("The topmost level (left) cannot be smaller than the bottommost level (right).")
		} else {
			showLevels(top, bottom);
		}
	})

	$("#filter-button").click(function() {
		var searchString = $("#search-input").val();
		$('#jstree').jstree('search', searchString);
	});

	$("#clear-button").click(function() {
		$('#jstree').jstree(true).clear_search();
		$("#search-input").val("");
	});

	$.tablesorter.themes.bootstrap = {
		table : 'table table-bordered table-hover',
		caption : 'caption',
		header : 'bootstrap-header', 
		sortNone : '',
		sortAsc : '',
		sortDesc : '',
		active : '', 
		hover : '', 
		icons : '', 
		iconSortNone : 'bootstrap-icon-unsorted',
		iconSortAsc : 'glyphicon glyphicon-chevron-up',
		iconSortDesc : 'glyphicon glyphicon-chevron-down',
		filterRow : '', 
		footerRow : '',
		footerCells : '',
		even : '', 
		odd : '' 
	};

//	$('#topic-modal').on('hidden.bs.modal', function (e) {
//    	$("#jstree").focus()
//    })
});
