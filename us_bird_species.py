#!/usr/bin/env python3
"""
Top 50 most common bird species in North America.
Based on eBird checklist frequency data (2024).

Contains:
- English name
- Scientific name (for Xeno-canto queries)
- Directory name (for file storage)
- Priority (1=very common, 2=common, 3=regular)
"""

# Format: (English name, Scientific name, directory_name, priority)
# Priority 1: Very common (>20% eBird frequency) - first training batch
# Priority 2: Common (15-20%) - second batch
# Priority 3: Regular (5-15%) - third batch

US_BIRD_SPECIES = [
    # ============================================================
    # PRIORITY 1: VERY COMMON SPECIES (>20% eBird frequency)
    # ============================================================

    # Doves
    ("Mourning Dove", "Zenaida macroura", "mourning_dove", 1),

    # Cardinals & Grosbeaks
    ("Northern Cardinal", "Cardinalis cardinalis", "northern_cardinal", 1),

    # Thrushes
    ("American Robin", "Turdus migratorius", "american_robin", 1),

    # Corvids
    ("American Crow", "Corvus brachyrhynchos", "american_crow", 1),
    ("Blue Jay", "Cyanocitta cristata", "blue_jay", 1),

    # Sparrows
    ("Song Sparrow", "Melospiza melodia", "song_sparrow", 1),

    # Blackbirds
    ("Red-winged Blackbird", "Agelaius phoeniceus", "red_winged_blackbird", 1),

    # Starlings
    ("European Starling", "Sturnus vulgaris", "european_starling", 1),

    # Finches
    ("American Goldfinch", "Spinus tristis", "american_goldfinch", 1),
    ("House Finch", "Haemorhous mexicanus", "house_finch", 1),

    # Waterfowl
    ("Canada Goose", "Branta canadensis", "canada_goose", 1),
    ("Mallard", "Anas platyrhynchos", "mallard", 1),

    # Woodpeckers
    ("Downy Woodpecker", "Dryobates pubescens", "downy_woodpecker", 1),

    # ============================================================
    # PRIORITY 2: COMMON SPECIES (15-20% eBird frequency)
    # ============================================================

    # Woodpeckers
    ("Red-bellied Woodpecker", "Melanerpes carolinus", "red_bellied_woodpecker", 2),
    ("Northern Flicker", "Colaptes auratus", "northern_flicker", 2),

    # Sparrows
    ("House Sparrow", "Passer domesticus", "house_sparrow", 2),

    # Vultures
    ("Turkey Vulture", "Cathartes aura", "turkey_vulture", 2),

    # Chickadees & Titmice
    ("Black-capped Chickadee", "Poecile atricapillus", "black_capped_chickadee", 2),
    ("Tufted Titmouse", "Baeolophus bicolor", "tufted_titmouse", 2),

    # Juncos
    ("Dark-eyed Junco", "Junco hyemalis", "dark_eyed_junco", 2),

    # Nuthatches
    ("White-breasted Nuthatch", "Sitta carolinensis", "white_breasted_nuthatch", 2),

    # Herons
    ("Great Blue Heron", "Ardea herodias", "great_blue_heron", 2),

    # Mimids
    ("Northern Mockingbird", "Mimus polyglottos", "northern_mockingbird", 2),

    # Wrens
    ("Carolina Wren", "Thryothorus ludovicianus", "carolina_wren", 2),

    # Hawks
    ("Red-tailed Hawk", "Buteo jamaicensis", "red_tailed_hawk", 2),

    # Grackles
    ("Common Grackle", "Quiscalus quiscula", "common_grackle", 2),

    # ============================================================
    # PRIORITY 3: REGULAR SPECIES (5-15% eBird frequency)
    # ============================================================

    # Swallows
    ("Barn Swallow", "Hirundo rustica", "barn_swallow", 3),
    ("Tree Swallow", "Tachycineta bicolor", "tree_swallow", 3),

    # Warblers
    ("Yellow-rumped Warbler", "Setophaga coronata", "yellow_rumped_warbler", 3),
    ("Common Yellowthroat", "Geothlypis trichas", "common_yellowthroat", 3),
    ("Pine Warbler", "Setophaga pinus", "pine_warbler", 3),

    # Gulls
    ("Ring-billed Gull", "Larus delawarensis", "ring_billed_gull", 3),

    # Mimids
    ("Gray Catbird", "Dumetella carolinensis", "gray_catbird", 3),
    ("Brown Thrasher", "Toxostoma rufum", "brown_thrasher", 3),

    # Blackbirds
    ("Brown-headed Cowbird", "Molothrus ater", "brown_headed_cowbird", 3),

    # Sparrows
    ("Chipping Sparrow", "Spizella passerina", "chipping_sparrow", 3),
    ("White-throated Sparrow", "Zonotrichia albicollis", "white_throated_sparrow", 3),

    # Bluebirds
    ("Eastern Bluebird", "Sialia sialis", "eastern_bluebird", 3),

    # Plovers
    ("Killdeer", "Charadrius vociferus", "killdeer", 3),

    # Flycatchers
    ("Eastern Phoebe", "Sayornis phoebe", "eastern_phoebe", 3),

    # Waxwings
    ("Cedar Waxwing", "Bombycilla cedrorum", "cedar_waxwing", 3),

    # Hummingbirds
    ("Ruby-throated Hummingbird", "Archilochus colubris", "ruby_throated_hummingbird", 3),

    # Orioles
    ("Baltimore Oriole", "Icterus galbula", "baltimore_oriole", 3),

    # Buntings
    ("Indigo Bunting", "Passerina cyanea", "indigo_bunting", 3),

    # Towhees
    ("Eastern Towhee", "Pipilo erythrophthalmus", "eastern_towhee", 3),

    # Finches
    ("Purple Finch", "Haemorhous purpureus", "purple_finch", 3),

    # Chickadees
    ("Carolina Chickadee", "Poecile carolinensis", "carolina_chickadee", 3),

    # Meadowlarks
    ("Eastern Meadowlark", "Sturnella magna", "eastern_meadowlark", 3),

    # Thrushes
    ("Wood Thrush", "Hylocichla mustelina", "wood_thrush", 3),

    # Tanagers
    ("Scarlet Tanager", "Piranga olivacea", "scarlet_tanager", 3),
]


def get_species_by_priority(priority: int = None) -> list:
    """Get species, optionally filtered by priority."""
    if priority is None:
        return US_BIRD_SPECIES
    return [s for s in US_BIRD_SPECIES if s[3] == priority]


def get_all_species_for_training() -> list:
    """Get all species in training order (priority 1 first)."""
    sorted_species = sorted(US_BIRD_SPECIES, key=lambda x: (x[3], x[0]))
    return [(s[0], s[1], s[2]) for s in sorted_species]


def count_species() -> dict:
    """Count species per priority."""
    counts = {1: 0, 2: 0, 3: 0}
    for s in US_BIRD_SPECIES:
        counts[s[3]] += 1
    counts['total'] = len(US_BIRD_SPECIES)
    return counts


if __name__ == "__main__":
    counts = count_species()
    print(f"\nNorth American Bird Species for Training")
    print("=" * 50)
    print(f"Priority 1 (very common):  {counts[1]} species")
    print(f"Priority 2 (common):       {counts[2]} species")
    print(f"Priority 3 (regular):      {counts[3]} species")
    print("-" * 50)
    print(f"TOTAL: {counts['total']} species")
