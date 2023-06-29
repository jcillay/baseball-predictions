""" """
# Calculate Weighted Runs Created Plus (wRC+)
def calculate_wRC_plus(hits, walks, hit_by_pitch, total_bases, plate_appearances, league_wRC_plus, league_runs):
    # Calculate components of wRC
    wOBA = ((0.72 * walks) + (0.75 * hit_by_pitch) + (0.90 * hits) + (0.92 * total_bases)) / plate_appearances
    league_plate_appearances = 1
    league_wOBA = league_runs / league_plate_appearances

    # Calculate wRC
    wRC = (wOBA - league_wOBA) * plate_appearances

    # Calculate wRC+
    wRC_plus = (wRC / plate_appearances) * 100

    return wRC_plus

# Example usage
hits = 140
walks = 50
hit_by_pitch = 5
total_bases = 220
plate_appearances = 400
league_wRC_plus = 100
league_runs = 5000

wRC_plus = calculate_wRC_plus(hits, walks, hit_by_pitch, total_bases, plate_appearances, league_wRC_plus, league_runs)

# Calculate Isolated Power (ISO)
def calculate_iso(slugging_percentage, batting_average):
    ISO = slugging_percentage - batting_average
    return ISO

# Example usage
slugging_percentage = 0.500
batting_average = 0.300

ISO = calculate_iso(slugging_percentage, batting_average)
