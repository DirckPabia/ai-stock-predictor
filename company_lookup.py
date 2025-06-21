def get_ph_companies():
    """
    Returns a dictionary of Philippine companies and their PSE symbols.
    This could later be replaced with a live PSE scraper or CSV loader.
    """
    return {
        "Jollibee Foods Corporation": "JFC.PS",
        "Ayala Land, Inc.": "ALI.PS",
        "SM Prime Holdings, Inc.": "SMPH.PS",
        "BDO Unibank, Inc.": "BDO.PS",
        "Ayala Corporation": "AC.PS",
        "Globe Telecom, Inc.": "GLO.PS",
        "PLDT Inc.": "TEL.PS",
        "Universal Robina Corporation": "URC.PS",
        "Manila Electric Company": "MER.PS"
    }

def get_global_companies():
    """
    Returns a dictionary of US/global companies and their ticker symbols.
    Later, this can read from a real-time data source or external CSV/API.
    """
    return {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "NVIDIA Corporation": "NVDA",
        "JPMorgan Chase & Co.": "JPM",
        "Bank of America Corporation": "BAC",
        "Exxon Mobil Corporation": "XOM",
        "Tesla, Inc.": "TSLA",
        "Amazon.com, Inc.": "AMZN"
    }
