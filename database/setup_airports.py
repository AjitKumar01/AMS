import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'airports.db')

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS airports (
        iata_code TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT NOT NULL,
        country TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL
    )
    ''')
    
    # Expanded list of major international airports
    airports = [
        # --- North America ---
        # USA
        ('JFK', 'John F. Kennedy International Airport', 'New York', 'USA', 40.6413, -73.7781),
        ('LAX', 'Los Angeles International Airport', 'Los Angeles', 'USA', 33.9416, -118.4085),
        ('ORD', 'O\'Hare International Airport', 'Chicago', 'USA', 41.9742, -87.9073),
        ('SFO', 'San Francisco International Airport', 'San Francisco', 'USA', 37.6213, -122.3790),
        ('ATL', 'Hartsfield-Jackson Atlanta International Airport', 'Atlanta', 'USA', 33.6407, -84.4277),
        ('DFW', 'Dallas/Fort Worth International Airport', 'Dallas', 'USA', 32.8998, -97.0403),
        ('DEN', 'Denver International Airport', 'Denver', 'USA', 39.8561, -104.6737),
        ('MIA', 'Miami International Airport', 'Miami', 'USA', 25.7959, -80.2870),
        ('SEA', 'Seattle-Tacoma International Airport', 'Seattle', 'USA', 47.4502, -122.3088),
        ('BOS', 'Logan International Airport', 'Boston', 'USA', 42.3656, -71.0096),
        ('MCO', 'Orlando International Airport', 'Orlando', 'USA', 28.4312, -81.3081),
        ('LAS', 'Harry Reid International Airport', 'Las Vegas', 'USA', 36.0840, -115.1537),
        ('CLT', 'Charlotte Douglas International Airport', 'Charlotte', 'USA', 35.2144, -80.9473),
        ('EWR', 'Newark Liberty International Airport', 'Newark', 'USA', 40.6895, -74.1745),
        ('PHX', 'Phoenix Sky Harbor International Airport', 'Phoenix', 'USA', 33.4342, -112.0116),
        ('IAH', 'George Bush Intercontinental Airport', 'Houston', 'USA', 29.9902, -95.3368),
        ('IAD', 'Dulles International Airport', 'Washington D.C.', 'USA', 38.9531, -77.4565),
        ('DTW', 'Detroit Metropolitan Airport', 'Detroit', 'USA', 42.2121, -83.3533),
        ('MSP', 'Minneapolis–Saint Paul International Airport', 'Minneapolis', 'USA', 44.8848, -93.2223),
        ('PHL', 'Philadelphia International Airport', 'Philadelphia', 'USA', 39.8729, -75.2437),
        
        # Canada
        ('YYZ', 'Toronto Pearson International Airport', 'Toronto', 'Canada', 43.6777, -79.6248),
        ('YVR', 'Vancouver International Airport', 'Vancouver', 'Canada', 49.1947, -123.1792),
        ('YUL', 'Montréal-Pierre Elliott Trudeau International Airport', 'Montreal', 'Canada', 45.4657, -73.7455),
        ('YYC', 'Calgary International Airport', 'Calgary', 'Canada', 51.1215, -114.0076),
        
        # Mexico
        ('MEX', 'Mexico City International Airport', 'Mexico City', 'Mexico', 19.4361, -99.0719),
        ('CUN', 'Cancún International Airport', 'Cancún', 'Mexico', 21.0365, -86.8771),
        ('GDL', 'Guadalajara International Airport', 'Guadalajara', 'Mexico', 20.5218, -103.3112),

        # --- Europe ---
        # UK
        ('LHR', 'Heathrow Airport', 'London', 'UK', 51.4700, -0.4543),
        ('LGW', 'Gatwick Airport', 'London', 'UK', 51.1537, -0.1821),
        ('MAN', 'Manchester Airport', 'Manchester', 'UK', 53.3537, -2.2750),
        ('EDI', 'Edinburgh Airport', 'Edinburgh', 'UK', 55.9508, -3.3615),
        
        # France
        ('CDG', 'Charles de Gaulle Airport', 'Paris', 'France', 49.0097, 2.5479),
        ('ORY', 'Orly Airport', 'Paris', 'France', 48.7262, 2.3652),
        ('NCE', 'Nice Côte d\'Azur Airport', 'Nice', 'France', 43.6633, 7.2167),
        
        # Germany
        ('FRA', 'Frankfurt Airport', 'Frankfurt', 'Germany', 50.0379, 8.5622),
        ('MUC', 'Munich Airport', 'Munich', 'Germany', 48.3536, 11.7750),
        ('BER', 'Berlin Brandenburg Airport', 'Berlin', 'Germany', 52.3667, 13.5033),
        ('HAM', 'Hamburg Airport', 'Hamburg', 'Germany', 53.6303, 9.9883),
        
        # Spain
        ('MAD', 'Adolfo Suárez Madrid–Barajas Airport', 'Madrid', 'Spain', 40.4839, -3.5679),
        ('BCN', 'Josep Tarradellas Barcelona-El Prat Airport', 'Barcelona', 'Spain', 41.2974, 2.0833),
        ('PMI', 'Palma de Mallorca Airport', 'Palma de Mallorca', 'Spain', 39.5517, 2.7388),
        
        # Italy
        ('FCO', 'Leonardo da Vinci–Fiumicino Airport', 'Rome', 'Italy', 41.8003, 12.2389),
        ('MXP', 'Malpensa Airport', 'Milan', 'Italy', 45.6301, 8.7255),
        ('VCE', 'Venice Marco Polo Airport', 'Venice', 'Italy', 45.5051, 12.3518),
        
        # Others
        ('AMS', 'Amsterdam Airport Schiphol', 'Amsterdam', 'Netherlands', 52.3105, 4.7683),
        ('ZRH', 'Zurich Airport', 'Zurich', 'Switzerland', 47.4582, 8.5555),
        ('GVA', 'Geneva Airport', 'Geneva', 'Switzerland', 46.2370, 6.1092),
        ('VIE', 'Vienna International Airport', 'Vienna', 'Austria', 48.1103, 16.5697),
        ('BRU', 'Brussels Airport', 'Brussels', 'Belgium', 50.9014, 4.4844),
        ('CPH', 'Copenhagen Airport', 'Copenhagen', 'Denmark', 55.6180, 12.6508),
        ('OSL', 'Oslo Airport', 'Oslo', 'Norway', 60.1976, 11.1004),
        ('ARN', 'Stockholm Arlanda Airport', 'Stockholm', 'Sweden', 59.6498, 17.9238),
        ('HEL', 'Helsinki Airport', 'Helsinki', 'Finland', 60.3172, 24.9633),
        ('DUB', 'Dublin Airport', 'Dublin', 'Ireland', 53.4264, -6.2499),
        ('LIS', 'Humberto Delgado Airport', 'Lisbon', 'Portugal', 38.7756, -9.1354),
        ('ATH', 'Athens International Airport', 'Athens', 'Greece', 37.9364, 23.9445),
        ('IST', 'Istanbul Airport', 'Istanbul', 'Turkey', 41.2753, 28.7519),
        ('SVO', 'Sheremetyevo International Airport', 'Moscow', 'Russia', 55.9726, 37.4146),
        ('DME', 'Domodedovo International Airport', 'Moscow', 'Russia', 55.4103, 37.9025),
        ('WAW', 'Warsaw Chopin Airport', 'Warsaw', 'Poland', 52.1672, 20.9679),
        ('PRG', 'Václav Havel Airport Prague', 'Prague', 'Czech Republic', 50.1008, 14.2600),
        ('BUD', 'Budapest Ferenc Liszt International Airport', 'Budapest', 'Hungary', 47.4385, 19.2523),

        # --- Asia ---
        # Japan
        ('HND', 'Tokyo Haneda Airport', 'Tokyo', 'Japan', 35.5494, 139.7798),
        ('NRT', 'Narita International Airport', 'Tokyo', 'Japan', 35.7719, 140.3929),
        ('KIX', 'Kansai International Airport', 'Osaka', 'Japan', 34.4320, 135.2304),
        ('ITM', 'Itami Airport', 'Osaka', 'Japan', 34.7855, 135.4382),
        ('NGO', 'Chubu Centrair International Airport', 'Nagoya', 'Japan', 34.8584, 136.8053),
        ('FUK', 'Fukuoka Airport', 'Fukuoka', 'Japan', 33.5859, 130.4506),
        ('CTS', 'New Chitose Airport', 'Sapporo', 'Japan', 42.7752, 141.6923),
        
        # China
        ('PEK', 'Beijing Capital International Airport', 'Beijing', 'China', 40.0799, 116.6031),
        ('PKX', 'Beijing Daxing International Airport', 'Beijing', 'China', 39.5092, 116.4106),
        ('PVG', 'Shanghai Pudong International Airport', 'Shanghai', 'China', 31.1443, 121.8083),
        ('SHA', 'Shanghai Hongqiao International Airport', 'Shanghai', 'China', 31.1979, 121.3363),
        ('CAN', 'Guangzhou Baiyun International Airport', 'Guangzhou', 'China', 23.3959, 113.2988),
        ('SZX', 'Shenzhen Bao\'an International Airport', 'Shenzhen', 'China', 22.6393, 113.8107),
        ('CTU', 'Chengdu Shuangliu International Airport', 'Chengdu', 'China', 30.5785, 103.9471),
        ('KMG', 'Kunming Changshui International Airport', 'Kunming', 'China', 25.1019, 102.9293),
        ('XIY', 'Xi\'an Xianyang International Airport', 'Xi\'an', 'China', 34.4471, 108.7516),
        
        # India
        ('DEL', 'Indira Gandhi International Airport', 'New Delhi', 'India', 28.5562, 77.1000),
        ('BOM', 'Chhatrapati Shivaji Maharaj International Airport', 'Mumbai', 'India', 19.0896, 72.8656),
        ('BLR', 'Kempegowda International Airport', 'Bangalore', 'India', 13.1986, 77.7066),
        ('MAA', 'Chennai International Airport', 'Chennai', 'India', 12.9941, 80.1709),
        ('HYD', 'Rajiv Gandhi International Airport', 'Hyderabad', 'India', 17.2403, 78.4294),
        ('CCU', 'Netaji Subhas Chandra Bose International Airport', 'Kolkata', 'India', 22.6547, 88.4467),
        
        # Southeast Asia
        ('SIN', 'Singapore Changi Airport', 'Singapore', 'Singapore', 1.3644, 103.9915),
        ('BKK', 'Suvarnabhumi Airport', 'Bangkok', 'Thailand', 13.6900, 100.7501),
        ('DMK', 'Don Mueang International Airport', 'Bangkok', 'Thailand', 13.9126, 100.6067),
        ('HKT', 'Phuket International Airport', 'Phuket', 'Thailand', 8.1132, 98.3169),
        ('KUL', 'Kuala Lumpur International Airport', 'Kuala Lumpur', 'Malaysia', 2.7456, 101.7099),
        ('CGK', 'Soekarno–Hatta International Airport', 'Jakarta', 'Indonesia', -6.1275, 106.6537),
        ('DPS', 'I Gusti Ngurah Rai International Airport', 'Denpasar', 'Indonesia', -8.7482, 115.1672),
        ('SGN', 'Tan Son Nhat International Airport', 'Ho Chi Minh City', 'Vietnam', 10.8188, 106.6519),
        ('HAN', 'Noi Bai International Airport', 'Hanoi', 'Vietnam', 21.2212, 105.8072),
        ('MNL', 'Ninoy Aquino International Airport', 'Manila', 'Philippines', 14.5086, 121.0194),
        
        # East Asia (Others)
        ('HKG', 'Hong Kong International Airport', 'Hong Kong', 'Hong Kong', 22.3080, 113.9185),
        ('ICN', 'Incheon International Airport', 'Seoul', 'South Korea', 37.4602, 126.4407),
        ('GMP', 'Gimpo International Airport', 'Seoul', 'South Korea', 37.5583, 126.7906),
        ('TPE', 'Taoyuan International Airport', 'Taipei', 'Taiwan', 25.0797, 121.2342),
        
        # --- Middle East ---
        ('DXB', 'Dubai International Airport', 'Dubai', 'UAE', 25.2532, 55.3657),
        ('DOH', 'Hamad International Airport', 'Doha', 'Qatar', 25.2609, 51.6138),
        ('AUH', 'Zayed International Airport', 'Abu Dhabi', 'UAE', 24.4441, 54.6511),
        ('JED', 'King Abdulaziz International Airport', 'Jeddah', 'Saudi Arabia', 21.6796, 39.1565),
        ('RUH', 'King Khalid International Airport', 'Riyadh', 'Saudi Arabia', 24.9576, 46.6988),
        ('TLV', 'Ben Gurion Airport', 'Tel Aviv', 'Israel', 32.0055, 34.8854),
        ('AMM', 'Queen Alia International Airport', 'Amman', 'Jordan', 31.7225, 35.9932),
        ('MCT', 'Muscat International Airport', 'Muscat', 'Oman', 23.5933, 58.2844),
        ('KWI', 'Kuwait International Airport', 'Kuwait City', 'Kuwait', 29.2266, 47.9689),
        ('BAH', 'Bahrain International Airport', 'Manama', 'Bahrain', 26.2708, 50.6336),
        
        # --- Oceania ---
        ('SYD', 'Sydney Kingsford Smith Airport', 'Sydney', 'Australia', -33.9399, 151.1753),
        ('MEL', 'Melbourne Airport', 'Melbourne', 'Australia', -37.6690, 144.8410),
        ('BNE', 'Brisbane Airport', 'Brisbane', 'Australia', -27.3842, 153.1175),
        ('PER', 'Perth Airport', 'Perth', 'Australia', -31.9385, 115.9672),
        ('AKL', 'Auckland Airport', 'Auckland', 'New Zealand', -37.0082, 174.7850),
        ('CHC', 'Christchurch Airport', 'Christchurch', 'New Zealand', -43.4894, 172.5323),
        
        # --- South America ---
        ('GRU', 'São Paulo/Guarulhos International Airport', 'São Paulo', 'Brazil', -23.4356, -46.4731),
        ('GIG', 'Rio de Janeiro/Galeão International Airport', 'Rio de Janeiro', 'Brazil', -22.8089, -43.2436),
        ('BOG', 'El Dorado International Airport', 'Bogotá', 'Colombia', 4.7016, -74.1469),
        ('LIM', 'Jorge Chávez International Airport', 'Lima', 'Peru', -12.0241, -77.1120),
        ('SCL', 'Arturo Merino Benítez International Airport', 'Santiago', 'Chile', -33.3930, -70.7858),
        ('EZE', 'Ministro Pistarini International Airport', 'Buenos Aires', 'Argentina', -34.8150, -58.5348),
        ('AEP', 'Jorge Newbery Airfield', 'Buenos Aires', 'Argentina', -34.5592, -58.4156),
        ('PTY', 'Tocumen International Airport', 'Panama City', 'Panama', 9.0714, -79.3835),
        
        # --- Africa ---
        ('JNB', 'O. R. Tambo International Airport', 'Johannesburg', 'South Africa', -26.1367, 28.2411),
        ('CPT', 'Cape Town International Airport', 'Cape Town', 'South Africa', -33.9715, 18.6021),
        ('CAI', 'Cairo International Airport', 'Cairo', 'Egypt', 30.1219, 31.4056),
        ('CMN', 'Mohammed V International Airport', 'Casablanca', 'Morocco', 33.3675, -7.5899),
        ('ADD', 'Bole International Airport', 'Addis Ababa', 'Ethiopia', 8.9779, 38.7993),
        ('NBO', 'Jomo Kenyatta International Airport', 'Nairobi', 'Kenya', -1.3192, 36.9275),
        ('LOS', 'Murtala Muhammed International Airport', 'Lagos', 'Nigeria', 6.5774, 3.3215),
        ('ACC', 'Kotoka International Airport', 'Accra', 'Ghana', 5.6052, -0.1668),
        ('TUN', 'Tunis–Carthage International Airport', 'Tunis', 'Tunisia', 36.8510, 10.2272),
        ('ALG', 'Houari Boumediene Airport', 'Algiers', 'Algeria', 36.6910, 3.2154)
    ]
    
    cursor.executemany('INSERT INTO airports VALUES (?, ?, ?, ?, ?, ?)', airports)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH} with {len(airports)} airports.")

if __name__ == "__main__":
    init_db()