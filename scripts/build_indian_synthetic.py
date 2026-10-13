"""Build a synthetic, fully annotated Indian clinical-text de-identification dataset.

Ten document types modelled on public Indian formats:
  * the IRDAI standardised discharge summary,
  * NABH and CAHO discharge-summary checklists,
  * common hospital templates (UHID, IP No, DOA/DOD, consultant, Father/Guardian, Address),
  * government "discharge tickets" (CR No, Adm No, unit, referral from PHC/CHC).

Every PHI value is inserted through `Doc.add`, which records its exact offsets. The labels follow
docs/10_indian_synthetic_annotation_guidelines.md (frozen before any evaluation).

    python scripts/build_indian_synthetic.py            -> data/synthetic_in/{dev,test}.jsonl + QA

Everything is fictional. E-mails use example.com/.org/.net, and numbers are random.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = random.Random(0)          # re-seeded per split in main()


def pick(seq):
    return R.choice(seq)


def chance(p: float) -> bool:
    return R.random() < p


def digits(n: int, first: str = "0123456789") -> str:
    return R.choice(first) + "".join(R.choice("0123456789") for _ in range(n - 1))


def letters(n: int) -> str:
    return "".join(R.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(n))


# ======================================================================= document builder
@dataclass(frozen=True)
class P:
    """A PHI value: inserted text + category."""
    text: str
    label: str


class Doc:
    def __init__(self, doc_type: str):
        self.doc_type = doc_type
        self.buf: list[str] = []
        self.n = 0
        self.spans: list[dict] = []

    def _w(self, s: str) -> None:
        self.buf.append(s)
        self.n += len(s)

    def add(self, *parts) -> "Doc":
        for x in parts:
            if x is None or x == "":
                continue
            if isinstance(x, P):
                self.spans.append({"start": self.n, "end": self.n + len(x.text), "label": x.label})
                self._w(x.text)
            elif isinstance(x, (list, tuple)):
                self.add(*x)
            else:
                self._w(str(x))
        return self

    def put(self, tpl: str, **kv) -> "Doc":
        """Template with {name} slots; slot values may be str, P or lists of both."""
        for tok in re.split(r"(\{[a-z_0-9]+\})", tpl):
            if re.fullmatch(r"\{[a-z_0-9]+\}", tok):
                self.add(kv[tok[1:-1]])
            elif tok:
                self._w(tok)
        return self

    @property
    def text(self) -> str:
        return "".join(self.buf)


# ======================================================================= value pools
REGIONS = {
    "telugu": dict(
        m=["Srinivas", "Ramesh", "Suresh", "Venkatesh", "Naresh", "Mahesh", "Prasad", "Krishna Mohan", "Satyanarayana",
           "Nagarjuna", "Kiran", "Chaitanya", "Harish", "Ravi Kumar", "Raju", "Venkata Ramana", "Anjaneyulu", "Sai Kiran"],
        f=["Lakshmi", "Padma", "Swathi", "Sravani", "Anusha", "Bhavani", "Sujatha", "Madhavi", "Kavitha", "Sirisha",
           "Rama Devi", "Pavani", "Vijaya Lakshmi", "Saritha", "Haritha"],
        sur=["Reddy", "Rao", "Naidu", "Chowdary", "Varma", "Goud", "Yadav", "Murthy", "Prasad", "Raju", "Setty", "Babu"],
        initial=True, cities=["Hyderabad", "Warangal", "Vijayawada", "Guntur", "Visakhapatnam", "Tirupati", "Karimnagar"]),
    "tamil": dict(
        m=["Murugan", "Senthil Kumar", "Karthik", "Saravanan", "Rajesh", "Arun", "Vignesh", "Balaji", "Ganesan",
           "Ramasamy", "Selvam", "Prabhu", "Dinesh", "Kannan"],
        f=["Meena", "Priya", "Revathi", "Kalaiselvi", "Divya", "Saranya", "Valli", "Anitha", "Malathi", "Jayanthi",
           "Kowsalya", "Sumathi"],
        sur=["Pillai", "Iyer", "Mudaliar", "Nadar", "Gounder", "Raman", "Krishnan", "Subramanian", "Natarajan",
             "Sundaram", "Rajendran"],
        initial=True, cities=["Chennai", "Madurai", "Coimbatore", "Tiruchirappalli", "Salem"]),
    "kannada": dict(
        m=["Manjunath", "Raghavendra", "Shivakumar", "Basavaraj", "Prakash", "Girish", "Nagaraj", "Mallikarjun",
           "Sudeep", "Harsha"],
        f=["Shilpa", "Pooja", "Rekha", "Savitha", "Roopa", "Asha", "Shwetha", "Bhagya", "Nandini", "Geetha"],
        sur=["Gowda", "Shetty", "Hegde", "Bhat", "Patil", "Kulkarni", "Naik", "Hiremath", "Kamath"],
        initial=True, cities=["Bengaluru", "Mysuru", "Mangaluru", "Hubballi"]),
    "malayalam": dict(
        m=["Rajeev", "Anil Kumar", "Sajeev", "Biju", "Shaji", "Unnikrishnan", "Vinod", "Jose", "Sebastian", "Ajith"],
        f=["Sreeja", "Bindu", "Remya", "Smitha", "Lekha", "Anju", "Jincy", "Sheeba", "Deepthi", "Ambily"],
        sur=["Nair", "Menon", "Kurup", "Varghese", "Kurian", "Mathew", "Panicker", "Nambiar", "Chacko", "Pillai"],
        initial=False, cities=["Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"]),
    "marathi": dict(
        m=["Sachin", "Santosh", "Prashant", "Vikas", "Ganesh", "Amol", "Rahul", "Sandeep", "Dattatray", "Nitin",
           "Yogesh", "Mangesh"],
        f=["Sunita", "Vaishali", "Manisha", "Ashwini", "Sneha", "Pallavi", "Rohini", "Sangeeta", "Archana", "Jyoti"],
        sur=["Patil", "Jadhav", "Pawar", "Deshmukh", "Kulkarni", "Joshi", "Shinde", "Gaikwad", "Kale", "Chavan",
             "Bhosale", "Deshpande"],
        initial=False, cities=["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"]),
    "hindi": dict(
        m=["Ram Prasad", "Shyam Lal", "Rajesh", "Sanjay", "Amit", "Vijay", "Manoj", "Pankaj", "Deepak", "Rakesh",
           "Mukesh", "Sunil", "Ajay", "Arvind", "Ramesh Chandra", "Brijesh", "Dinesh", "Rohit", "Vinod"],
        f=["Sunita", "Geeta", "Rekha", "Pooja", "Neha", "Anita", "Kiran", "Savitri", "Kamla", "Shanti", "Priyanka",
           "Rita", "Usha", "Manju", "Pushpa", "Mamta"],
        sur=["Sharma", "Verma", "Gupta", "Singh", "Yadav", "Mishra", "Tiwari", "Pandey", "Srivastava", "Shukla",
             "Chauhan", "Agarwal", "Jain", "Maurya", "Kushwaha", "Prajapati", "Paswan", "Dubey", "Rastogi"],
        sur_f=["Devi", "Kumari"], sur_m=["Kumar"],
        initial=False, cities=["Lucknow", "New Delhi", "Patna", "Jaipur", "Varanasi", "Kanpur", "Indore", "Bhopal",
                               "Gorakhpur", "Ranchi"]),
    "bengali_odia": dict(
        m=["Subhash", "Sourav", "Debashis", "Partha", "Arindam", "Biswajit", "Prasanta", "Abhijit", "Sambit",
           "Pradeep", "Bikash"],
        f=["Mousumi", "Rina", "Tanushree", "Sharmila", "Moumita", "Rupa", "Sutapa", "Mamata", "Sasmita", "Puja"],
        sur=["Das", "Ghosh", "Bose", "Banerjee", "Chatterjee", "Mukherjee", "Sen", "Roy", "Dutta", "Mondal", "Sahoo",
             "Mohanty", "Behera", "Panda", "Barua", "Hazarika"],
        initial=False, cities=["Kolkata", "Bhubaneswar", "Cuttack", "Guwahati", "Siliguri"]),
    "gujarati": dict(
        m=["Hitesh", "Jignesh", "Bhavesh", "Ketan", "Paresh", "Nilesh", "Mehul", "Chirag"],
        f=["Hetal", "Komal", "Nisha", "Bhavna", "Dipti", "Kinjal", "Falguni"],
        sur=["Patel", "Shah", "Desai", "Mehta", "Parmar", "Trivedi", "Bhatt", "Solanki"],
        initial=False, cities=["Ahmedabad", "Surat", "Vadodara", "Rajkot"]),
    "punjabi": dict(
        m=["Gurpreet", "Harjeet", "Jaswinder", "Manpreet", "Balwinder", "Kuldeep", "Harpal", "Amarjit"],
        f=["Harjeet", "Manjeet", "Simran", "Gurleen", "Paramjit", "Jasleen"],
        sur=["Singh", "Singh Sandhu", "Singh Gill", "Singh Dhillon", "Singh Grewal", "Singh Brar"],
        sur_f_only=["Kaur", "Kaur Sidhu", "Kaur Gill", "Kaur Dhillon"],
        initial=False, cities=["Ludhiana", "Amritsar", "Chandigarh", "Jalandhar"]),
    "muslim": dict(
        m=["Mohammed Irfan", "Syed Ahmed", "Abdul Rahman", "Imran", "Salim", "Mohd. Arif", "Faisal", "Shahid",
           "Abdul Kareem", "Rizwan", "Mohammed Rafi"],
        f=["Fatima", "Ayesha", "Shabana", "Nazia", "Rukhsar", "Salma", "Farheen", "Zainab", "Shaheen"],
        sur=["Khan", "Ansari", "Shaikh", "Qureshi", "Siddiqui", "Pathan", "Hussain", "Mirza"],
        sur_f=["Begum", "Bano"],
        initial=False, cities=["Hyderabad", "Lucknow", "Bhopal", "Mumbai", "Kolkata", "Aligarh"]),
    "christian": dict(
        m=["Joseph", "Francis", "Anthony", "Peter", "Stephen", "Lawrence", "Wilson"],
        f=["Mary", "Elizabeth", "Jessy", "Shiny", "Merlin", "Agnes", "Rosamma"],
        sur=["D'Souza", "Fernandes", "Pinto", "Rodrigues", "Varghese", "Mathew", "Kurian", "Lobo", "Dsilva"],
        initial=False, cities=["Mangaluru", "Kochi", "Panaji", "Chennai", "Mumbai"]),
}
REGION_WEIGHTS = {"telugu": 14, "tamil": 10, "kannada": 8, "malayalam": 7, "marathi": 10, "hindi": 22,
                  "bengali_odia": 10, "gujarati": 5, "punjabi": 4, "muslim": 7, "christian": 3}

CITIES = {
    "Hyderabad": ("Telangana", "500", "040", ["Banjara Hills", "Kukatpally", "Ameerpet", "Dilsukhnagar", "LB Nagar",
                  "Mehdipatnam", "Madhapur", "Malakpet", "Uppal", "Begumpet"],
                  ["Shamshabad", "Chevella", "Ibrahimpatnam", "Medchal", "Ghatkesar", "Moinabad"]),
    "Warangal": ("Telangana", "506", "0870", ["Hanamkonda", "Kazipet", "Subedari"], ["Narsampet", "Parkal", "Jangaon"]),
    "Karimnagar": ("Telangana", "505", "0878", ["Mukarampura", "Jyothinagar"], ["Huzurabad", "Jagtial", "Peddapalli"]),
    "Vijayawada": ("Andhra Pradesh", "520", "0866", ["Benz Circle", "Patamata", "Governorpet"], ["Nuzvid", "Gudivada", "Kankipadu"]),
    "Guntur": ("Andhra Pradesh", "522", "0863", ["Brodipet", "Arundelpet", "Lakshmipuram"], ["Tenali", "Mangalagiri", "Narasaraopet"]),
    "Visakhapatnam": ("Andhra Pradesh", "530", "0891", ["MVP Colony", "Gajuwaka", "Dwaraka Nagar"], ["Anakapalli", "Bheemunipatnam"]),
    "Tirupati": ("Andhra Pradesh", "517", "0877", ["Tiruchanoor", "Balaji Colony"], ["Srikalahasti", "Puttur", "Chandragiri"]),
    "Chennai": ("Tamil Nadu", "600", "044", ["T. Nagar", "Adyar", "Anna Nagar", "Velachery", "Tambaram", "Mylapore", "Perambur"],
                ["Sriperumbudur", "Chengalpattu", "Ponneri", "Tiruvallur"]),
    "Madurai": ("Tamil Nadu", "625", "0452", ["K.K. Nagar", "Anna Nagar", "Tallakulam"], ["Melur", "Usilampatti", "Thirumangalam"]),
    "Coimbatore": ("Tamil Nadu", "641", "0422", ["RS Puram", "Gandhipuram", "Peelamedu"], ["Pollachi", "Mettupalayam", "Annur"]),
    "Tiruchirappalli": ("Tamil Nadu", "620", "0431", ["Srirangam", "Thillai Nagar"], ["Lalgudi", "Manapparai"]),
    "Salem": ("Tamil Nadu", "636", "0427", ["Fairlands", "Hasthampatti"], ["Attur", "Omalur", "Mettur"]),
    "Bengaluru": ("Karnataka", "560", "080", ["Jayanagar", "Indiranagar", "Whitefield", "Malleswaram", "Yelahanka",
                  "BTM Layout", "Koramangala", "Rajajinagar"], ["Hoskote", "Devanahalli", "Nelamangala", "Anekal"]),
    "Mysuru": ("Karnataka", "570", "0821", ["Kuvempunagar", "Vijayanagar", "Saraswathipuram"], ["Nanjangud", "Hunsur", "T. Narasipur"]),
    "Mangaluru": ("Karnataka", "575", "0824", ["Kadri", "Bejai", "Kankanady"], ["Bantwal", "Puttur", "Mulki"]),
    "Hubballi": ("Karnataka", "580", "0836", ["Vidyanagar", "Gokul Road"], ["Kundgol", "Navalgund"]),
    "Kochi": ("Kerala", "682", "0484", ["Edappally", "Kakkanad", "Vyttila", "Kaloor"], ["Aluva", "Perumbavoor", "Angamaly"]),
    "Thiruvananthapuram": ("Kerala", "695", "0471", ["Pattom", "Kazhakootam", "Vazhuthacaud"], ["Neyyattinkara", "Attingal", "Nedumangad"]),
    "Kozhikode": ("Kerala", "673", "0495", ["Nadakkavu", "Mavoor Road"], ["Vadakara", "Koyilandy", "Feroke"]),
    "Thrissur": ("Kerala", "680", "0487", ["Poothole", "Ayyanthole"], ["Chalakudy", "Kodungallur", "Irinjalakuda"]),
    "Mumbai": ("Maharashtra", "400", "022", ["Andheri (E)", "Borivali (W)", "Dadar", "Chembur", "Malad (W)", "Ghatkopar", "Kurla"],
               ["Vasai", "Bhiwandi", "Kalyan", "Panvel"]),
    "Pune": ("Maharashtra", "411", "020", ["Kothrud", "Hadapsar", "Aundh", "Pimpri", "Wakad", "Shivajinagar"],
             ["Shirur", "Baramati", "Wadgaon", "Haveli", "Junnar"]),
    "Nagpur": ("Maharashtra", "440", "0712", ["Dharampeth", "Sitabuldi", "Manish Nagar"], ["Katol", "Kamptee", "Umred"]),
    "Nashik": ("Maharashtra", "422", "0253", ["Gangapur Road", "Panchavati"], ["Sinnar", "Igatpuri", "Niphad"]),
    "Aurangabad": ("Maharashtra", "431", "0240", ["Cidco", "Garkheda"], ["Paithan", "Vaijapur", "Kannad"]),
    "Lucknow": ("Uttar Pradesh", "226", "0522", ["Gomti Nagar", "Aliganj", "Indira Nagar", "Alambagh", "Chinhat", "Rajajipuram"],
                ["Malihabad", "Mohanlalganj", "Bakshi Ka Talab", "Kakori", "Itaunja"]),
    "Varanasi": ("Uttar Pradesh", "221", "0542", ["Lanka", "Sigra", "Bhelupur"], ["Chaubeypur", "Pindra", "Rajatalab"]),
    "Kanpur": ("Uttar Pradesh", "208", "0512", ["Kidwai Nagar", "Swaroop Nagar", "Kalyanpur"], ["Bilhaur", "Ghatampur", "Bithoor"]),
    "Gorakhpur": ("Uttar Pradesh", "273", "0551", ["Golghar", "Shahpur"], ["Sahjanwa", "Chauri Chaura", "Pipraich"]),
    "Aligarh": ("Uttar Pradesh", "202", "0571", ["Civil Lines", "Dodhpur"], ["Khair", "Atrauli", "Iglas"]),
    "New Delhi": ("Delhi", "110", "011", ["Dwarka", "Rohini", "Lajpat Nagar", "Karol Bagh", "Mayur Vihar", "Janakpuri"],
                  ["Narela", "Najafgarh", "Alipur"]),
    "Patna": ("Bihar", "800", "0612", ["Kankarbagh", "Boring Road", "Rajendra Nagar", "Patliputra Colony"],
              ["Danapur", "Phulwari Sharif", "Masaurhi", "Bihta"]),
    "Jaipur": ("Rajasthan", "302", "0141", ["Malviya Nagar", "Vaishali Nagar", "Mansarovar", "Raja Park"], ["Chomu", "Sanganer", "Phagi", "Bassi"]),
    "Indore": ("Madhya Pradesh", "452", "0731", ["Vijay Nagar", "Palasia", "Rajendra Nagar"], ["Mhow", "Sanwer", "Depalpur"]),
    "Bhopal": ("Madhya Pradesh", "462", "0755", ["Arera Colony", "Kolar Road", "MP Nagar"], ["Berasia", "Sehore", "Obedullaganj"]),
    "Ranchi": ("Jharkhand", "834", "0651", ["Lalpur", "Harmu", "Kanke"], ["Ormanjhi", "Bundu", "Tamar"]),
    "Kolkata": ("West Bengal", "700", "033", ["Salt Lake", "Behala", "Garia", "Dum Dum", "Tollygunge", "Shyambazar"],
                ["Barasat", "Baruipur", "Diamond Harbour", "Barrackpore"]),
    "Siliguri": ("West Bengal", "734", "0353", ["Pradhan Nagar", "Hakimpara"], ["Matigara", "Naxalbari", "Kharibari"]),
    "Bhubaneswar": ("Odisha", "751", "0674", ["Saheed Nagar", "Patia", "Nayapalli", "Old Town"], ["Khurda", "Jatni", "Balianta"]),
    "Cuttack": ("Odisha", "753", "0671", ["Buxi Bazar", "Badambadi"], ["Athagarh", "Banki", "Salepur"]),
    "Guwahati": ("Assam", "781", "0361", ["Dispur", "Beltola", "Zoo Road", "Chandmari"], ["Mirza", "Sonapur", "Hajo"]),
    "Ahmedabad": ("Gujarat", "380", "079", ["Navrangpura", "Maninagar", "Satellite", "Vastrapur"], ["Sanand", "Dholka", "Viramgam"]),
    "Surat": ("Gujarat", "395", "0261", ["Adajan", "Varachha", "Athwa"], ["Bardoli", "Kamrej", "Olpad"]),
    "Vadodara": ("Gujarat", "390", "0265", ["Alkapuri", "Karelibaug"], ["Padra", "Savli", "Dabhoi"]),
    "Rajkot": ("Gujarat", "360", "0281", ["Kalawad Road", "Raiya Road"], ["Gondal", "Jetpur", "Dhoraji"]),
    "Ludhiana": ("Punjab", "141", "0161", ["Model Town", "Sarabha Nagar", "Dugri"], ["Khanna", "Samrala", "Jagraon"]),
    "Amritsar": ("Punjab", "143", "0183", ["Ranjit Avenue", "Lawrence Road"], ["Ajnala", "Baba Bakala", "Majitha"]),
    "Jalandhar": ("Punjab", "144", "0181", ["Model Town", "Urban Estate"], ["Nakodar", "Phillaur", "Shahkot"]),
    "Chandigarh": ("Chandigarh", "160", "0172", ["Sector 22", "Sector 35", "Sector 44"], ["Manimajra", "Kharar", "Zirakpur"]),
    "Panaji": ("Goa", "403", "0832", ["Altinho", "Miramar"], ["Mapusa", "Ponda", "Bicholim"]),
}

CITY_HOSP = {
    "Hyderabad": ["Yashoda Hospitals", "KIMS Hospitals", "Care Hospitals", "Medicover Hospitals", "Continental Hospitals",
                  "Rainbow Children's Hospital", "Star Hospitals", "Kamineni Hospitals", "Apollo Hospitals", "Sunshine Hospitals"],
    "Chennai": ["Apollo Hospitals", "MIOT International", "Kauvery Hospital", "Sri Ramachandra Medical Centre", "Gleneagles Global Health City"],
    "Bengaluru": ["Manipal Hospital", "Narayana Health City", "Aster CMI Hospital", "Sparsh Hospital", "Fortis Hospital", "St. John's Medical College Hospital"],
    "Mumbai": ["Jaslok Hospital", "Kokilaben Dhirubhai Ambani Hospital", "Lilavati Hospital", "Breach Candy Hospital", "Bombay Hospital", "Hinduja Hospital"],
    "Pune": ["Ruby Hall Clinic", "Sahyadri Hospital", "Jehangir Hospital", "Deenanath Mangeshkar Hospital"],
    "New Delhi": ["Sir Ganga Ram Hospital", "Max Super Speciality Hospital", "Indraprastha Apollo Hospital", "BLK-Max Hospital", "Fortis Escorts Heart Institute"],
    "Lucknow": ["Sahara Hospital", "Medanta Hospital", "Mayo Medical Centre", "Charak Hospital"],
    "Kolkata": ["AMRI Hospitals", "Peerless Hospital", "Ruby General Hospital", "Belle Vue Clinic", "Medica Superspecialty Hospital"],
    "Kochi": ["Amrita Hospital", "Lisie Hospital", "Lakeshore Hospital", "Rajagiri Hospital"],
    "Ahmedabad": ["Sterling Hospital", "Zydus Hospital", "Shalby Hospital", "SAL Hospital"],
    "Bhubaneswar": ["SUM Hospital", "Kalinga Hospital", "AMRI Hospitals"],
    "Patna": ["Paras HMRI Hospital", "Ruban Memorial Hospital"],
    "Jaipur": ["Fortis Escorts Hospital", "Mahatma Gandhi Hospital", "Eternal Hospital"],
    "Vijayawada": ["Ramesh Hospitals", "Andhra Hospitals"],
    "Coimbatore": ["KMCH", "PSG Hospitals", "Ganga Hospital"],
    "Ludhiana": ["Dayanand Medical College and Hospital", "Christian Medical College"],
    "Guwahati": ["GNRC Hospitals", "Nemcare Hospital"],
}
GOVT_NAMED = {
    "Hyderabad": ["Osmania General Hospital", "Gandhi Hospital", "Nizam's Institute of Medical Sciences", "Niloufer Hospital"],
    "Lucknow": ["King George's Medical University", "Sanjay Gandhi Postgraduate Institute of Medical Sciences",
                "Ram Manohar Lohia Institute of Medical Sciences", "Balrampur Hospital"],
    "New Delhi": ["AIIMS", "Safdarjung Hospital", "Lok Nayak Hospital", "GTB Hospital", "Ram Manohar Lohia Hospital"],
    "Chennai": ["Rajiv Gandhi Government General Hospital", "Stanley Medical College Hospital", "Kilpauk Medical College Hospital"],
    "Mumbai": ["KEM Hospital", "JJ Hospital", "Nair Hospital", "Tata Memorial Hospital"],
    "Kolkata": ["SSKM Hospital", "NRS Medical College", "RG Kar Medical College"],
    "Bengaluru": ["Victoria Hospital", "Bowring Hospital", "NIMHANS", "Kidwai Memorial Institute of Oncology"],
    "Patna": ["Patna Medical College Hospital", "IGIMS", "Nalanda Medical College Hospital"],
    "Chandigarh": ["PGIMER", "GMCH"],
    "Pune": ["Sassoon General Hospital"],
    "Jaipur": ["SMS Hospital"],
    "Bhubaneswar": ["AIIMS", "Capital Hospital"],
    "Cuttack": ["SCB Medical College"],
    "Varanasi": ["Sir Sunderlal Hospital"],
    "Gorakhpur": ["BRD Medical College"],
    "Guntur": ["Guntur Medical College"],
    "Madurai": ["Government Rajaji Hospital"],
}
LOCAL_HOSP_PREFIX = ["Sri Sai", "Lifeline", "Sanjeevani", "Shifa", "Sunrise", "Navjeevan", "Arogya", "Matru Seva",
                     "Shree Krishna", "Siddhi Vinayak", "Lotus", "Om Sai", "Jeevan Jyoti", "Amrutha", "Sri Venkateswara",
                     "Vardaan", "Surya", "Kalyani", "Annapurna", "Holy Family", "St. Mary's", "Sai Krupa", "Vatsalya",
                     "Ashwini", "Sri Lakshmi", "Dhanvantari", "Aarogya Sri", "Noble", "Hope"]
LOCAL_HOSP_SUFFIX = ["Hospital", "Multispeciality Hospital", "Nursing Home", "Clinic", "Children's Hospital",
                     "Maternity Home", "Heart Centre", "Superspeciality Hospital", "Hospitals", "Medical Centre"]
LABS = ["Vijaya Diagnostic Centre", "Thyrocare", "Dr. Lal PathLabs", "SRL Diagnostics", "Metropolis Healthcare",
        "Tenet Diagnostics", "Lucid Diagnostics", "Medall Diagnostics", "Sai Scans & Diagnostics", "Aarthi Scans",
        "Om Diagnostics", "Neuberg Diagnostics"]
GENERIC_FACILITY = ["Government General Hospital, {t}", "District Hospital, {t}", "Area Hospital, {t}", "CHC {t}",
                    "PHC {t}", "Community Health Centre, {t}", "Primary Health Centre, {t}", "Civil Hospital, {t}",
                    "Sub-District Hospital, {t}", "UPHC {t}"]
EMPLOYERS = ["Infosys", "Tata Steel", "Indian Railways", "State Bank of India", "Wipro", "BHEL", "Singareni Collieries",
             "Reliance Industries", "Larsen & Toubro", "Coal India", "TSRTC", "Hindustan Aeronautics", "Cognizant"]
OCCUPATIONS = ["farmer", "daily wage labourer", "auto-rickshaw driver", "homemaker", "school teacher", "shopkeeper",
               "retired government employee", "student", "construction worker", "tailor", "security guard", "vegetable vendor"]
STREETS = ["Gandhi Road", "MG Road", "Station Road", "Temple Street", "Main Road", "Ambedkar Colony", "Church Street",
           "Bazaar Street", "Nehru Street", "Netaji Road", "Shivaji Path", "Old Market Road"]
LANDMARKS = ["Hanuman Temple", "SBI Bank", "Govt. High School", "Bus Stand", "Water Tank", "Jama Masjid", "Railway Station",
             "Petrol Bunk", "Post Office", "Ration Shop", "Sai Baba Temple"]
BUILDINGS = ["Sai Residency", "Lakshmi Towers", "Green Park Apartments", "Sri Balaji Enclave", "Shanti Kunj",
             "Royal Heights", "Krishna Apartments", "Silver Oak Residency", "Om Shanti Apartments"]
DEPTS = {
    "General Medicine": ["MD (General Medicine)", "MD (Medicine)", "MBBS, MD", "DNB (General Medicine)"],
    "Cardiology": ["MD, DM (Cardiology)", "DM (Cardiology)", "MD, DNB (Cardiology)"],
    "Neurology": ["MD, DM (Neurology)", "DM (Neuro)"],
    "Nephrology": ["MD, DM (Nephrology)", "DNB (Nephrology)"],
    "General Surgery": ["MS (General Surgery)", "MS, FMAS", "MS, MCh (GI Surgery)"],
    "Orthopaedics": ["MS (Ortho)", "MS (Orthopaedics), DNB", "D.Ortho"],
    "Pulmonology": ["MD (Pulmonary Medicine)", "MD (Respiratory Medicine)"],
    "Paediatrics": ["MD (Paediatrics)", "DCH, DNB (Paed)", "MD (Paeds)"],
    "Obstetrics & Gynaecology": ["MS (OBG)", "DGO, DNB", "MD (OBG)"],
    "Gastroenterology": ["MD, DM (Gastroenterology)", "DM (Gastro)"],
    "Psychiatry": ["MD (Psychiatry)", "DPM"],
}
COUNCILS = ["TSMC", "APMC", "KMC", "TNMC", "MMC", "DMC", "UPMC", "WBMC", "OMC", "GMC", "PMC", "TCMC"]
STATE_COUNCIL = {"Telangana": "TSMC", "Andhra Pradesh": "APMC", "Karnataka": "KMC", "Tamil Nadu": "TNMC",
                 "Maharashtra": "MMC", "Delhi": "DMC", "Uttar Pradesh": "UPMC", "West Bengal": "WBMC", "Odisha": "OMC",
                 "Gujarat": "GMC", "Punjab": "PMC", "Kerala": "TCMC"}
VEH_STATE = {"Telangana": "TS", "Andhra Pradesh": "AP", "Karnataka": "KA", "Tamil Nadu": "TN", "Maharashtra": "MH",
             "Delhi": "DL", "Uttar Pradesh": "UP", "West Bengal": "WB", "Odisha": "OD", "Gujarat": "GJ", "Punjab": "PB",
             "Kerala": "KL", "Bihar": "BR", "Rajasthan": "RJ", "Madhya Pradesh": "MP", "Jharkhand": "JH", "Assam": "AS",
             "Chandigarh": "CH", "Goa": "GA"}
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

FEMALE_ONLY_SURNAMES = {s for r in REGIONS.values() for k in ("sur_f", "sur_f_only") for s in r.get(k, [])}

# Phrases that contain pool words but are NOT PHI (hard negatives, allowed by the QA scan).
HARD_NEGATIVE_PHRASES = ["Thomas splint", "ASHA worker", "ASHA", "Ganga Hospital", "Gandhi Road", "Nehru Street",
                         "Ambedkar Colony", "Shivaji Path", "Sai Baba Temple", "Netaji Road", "Krishna Apartments",
                         "Lakshmi Towers", "Sri Balaji Enclave", "Om Shanti Apartments", "Shanti Kunj"]


# ======================================================================= value generators
@dataclass
class Person:
    first: str
    last: str
    sex: str
    region: str

    def render(self, style: str = "full") -> P:
        f, l = self.first, self.last
        if style == "upper":
            s = f"{f} {l}".upper()
        elif style == "initial_first":
            s = f"{l[0]}. {f}"
        elif style == "initial_first_nodot":
            s = f"{l[0]} {f}"
        elif style == "first_initial":
            s = f"{f} {l[0]}."
        elif style == "first":
            s = f
        elif style == "last":
            s = l
        else:
            s = f"{f} {l}"
        return P(s, "PERSON")


def new_person(region: str | None = None, sex: str | None = None) -> Person:
    region = region or R.choices(list(REGION_WEIGHTS), weights=list(REGION_WEIGHTS.values()))[0]
    r = REGIONS[region]
    sex = sex or pick("MF")
    first = pick(r["m"] if sex == "M" else r["f"])
    if sex == "F" and r.get("sur_f_only"):
        last = pick(r["sur_f_only"])
    elif sex == "F" and r.get("sur_f") and chance(0.4):
        last = pick(r["sur_f"])
    elif sex == "M" and r.get("sur_m") and chance(0.3):
        last = pick(r["sur_m"])
    else:
        last = pick(r["sur"])
    return Person(first, last, sex, region)


def name_style(p: Person, header: bool = False) -> str:
    opts = ["full"] * 6 + (["upper"] * 3 if header else [])
    if REGIONS[p.region]["initial"]:
        opts += ["initial_first"] * 3 + ["initial_first_nodot"]
    opts += ["first_initial"]
    return pick(opts)


def title(p: Person, age: int, married: bool = True) -> str:
    if age < 1:
        return pick(["Baby ", "B/O "]) if False else "Baby "
    if p.sex == "M":
        return pick(["Master ", ""]) if age < 14 else pick(["Mr. ", "Mr ", "Shri ", "Sri ", "", "Mr. "])
    if age < 14:
        return pick(["Kum. ", "Baby ", "Miss ", ""])
    return pick(["Mrs. ", "Smt. ", "Mrs ", "Smt ", "", "Ms. "]) if married else pick(["Ms. ", "Kum. ", "Miss ", ""])


def doctor() -> Person:
    return new_person()


def doc_title(prof: bool = False) -> str:
    return pick(["Prof. ", "Prof. Dr. "]) if prof else pick(["Dr. ", "Dr. ", "Dr ", "DR. "])


def fmt_date(d: dt.date, style: str) -> str:
    dd, mm, y = d.day, d.month, d.year
    m3 = MON[mm - 1][:3]
    o = "th" if 11 <= dd % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(dd % 10, "th")
    return {
        "dmy_slash": f"{dd:02d}/{mm:02d}/{y}", "dmy_dash": f"{dd:02d}-{mm:02d}-{y}", "dmy_dot": f"{dd:02d}.{mm:02d}.{y}",
        "dmy_slash_np": f"{dd}/{mm}/{y}", "dmy_slash_yy": f"{dd:02d}/{mm:02d}/{y % 100:02d}",
        "dmony_dash": f"{dd:02d}-{m3}-{y}", "dmony_dash_up": f"{dd:02d}-{m3.upper()}-{y}", "dmony_sp": f"{dd:02d} {m3} {y}",
        "dmonthy": f"{dd} {MON[mm - 1]} {y}", "dthmonthy": f"{dd}{o} {MON[mm - 1]} {y}", "mdy_text": f"{MON[mm - 1]} {dd}, {y}",
        "iso": f"{y}-{mm:02d}-{dd:02d}", "dmony_slash": f"{dd:02d}/{m3}/{y}", "dmony_yy": f"{dd:02d}-{m3}-{y % 100:02d}",
    }[style]


DATE_STYLES = ["dmy_slash"] * 6 + ["dmy_dash"] * 4 + ["dmy_dot"] * 3 + ["dmony_dash"] * 3 + ["dmony_sp"] * 2 + \
              ["dmy_slash_np", "dmy_slash_yy", "dmony_dash_up", "dmonthy", "dthmonthy", "mdy_text", "iso", "dmony_slash",
               "dmony_yy"]


def short_date(d: dt.date) -> str:
    o = "th" if 11 <= d.day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return pick([f"{d.day}/{d.month}", f"{d.day}{o} {MON[d.month - 1]}", f"{d.day:02d} {MON[d.month - 1][:3]}",
                 f"{d.day}-{MON[d.month - 1][:3]}"])


def D(d: dt.date, style: str) -> P:
    return P(fmt_date(d, style), "DATE")


def fmt_time(h: int, m: int, style: str) -> str:
    h12 = h % 12 or 12
    ap = "AM" if h < 12 else "PM"
    return {
        "ampm": f"{h12}:{m:02d} {ap}", "ampm_lower": f"{h12}:{m:02d} {ap.lower()}", "ampm_nospace": f"{h12:02d}:{m:02d}{ap}",
        "dot_ampm": f"{h12}.{m:02d} {ap.lower()}", "dot_ampm_dots": f"{h12}.{m:02d} {ap[0].lower()}.m.",
        "hrs": f"{h:02d}{m:02d} hrs", "hm_hrs": f"{h:02d}:{m:02d} hrs", "hm24": f"{h:02d}:{m:02d}",
        "hms": f"{h:02d}:{m:02d}:{R.randint(0, 59):02d}", "h_ampm": f"{h12} {ap.lower()}",
    }[style]


TIME_STYLES = ["ampm"] * 4 + ["ampm_lower"] * 2 + ["hm24"] * 3 + ["hrs"] * 2 + ["hm_hrs", "dot_ampm", "dot_ampm_dots",
                                                                                 "ampm_nospace", "hms"]


def T(style: str | None = None, hour: int | None = None) -> P:
    h = R.randint(0, 23) if hour is None else hour
    m = pick([0, 5, 10, 15, 20, 30, 40, 45, 50]) if chance(0.8) else R.randint(0, 59)
    s = style or pick(TIME_STYLES)
    if s == "h_ampm":
        m = 0
    return P(fmt_time(h, m, s), "TIME")


def mobile(style: str | None = None) -> P:
    n = digits(10, "6789")
    s = style or pick(["plain"] * 4 + ["sp5"] * 3 + ["plus91_sp"] * 3 + ["plus91_dash", "plus91", "zero", "dash5", "433"])
    return P({"plain": n, "sp5": f"{n[:5]} {n[5:]}", "plus91_sp": f"+91 {n[:5]} {n[5:]}", "plus91_dash": f"+91-{n}",
              "plus91": f"+91{n}", "zero": f"0{n}", "dash5": f"{n[:5]}-{n[5:]}",
              "433": f"{n[:4]} {n[4:7]} {n[7:]}"}[s], "PHONE")


def landline(city: str) -> P:
    std = CITIES[city][2]
    local = digits(10 - (len(std) - 1), "23456")
    s = pick(["dash", "dash", "paren", "space", "plus91"])
    if s == "dash":
        v = f"{std}-{local}"
    elif s == "paren":
        v = f"({std}) {local[:len(local) - 4]} {local[-4:]}"
    elif s == "space":
        v = f"{std} {local}"
    else:
        v = f"+91 {std[1:]} {local}"
    return P(v, "PHONE")


def tollfree() -> P:
    return P(pick([f"1800 {digits(3)} {digits(4)}", f"1800-{digits(3)}-{digits(4)}", f"1860 {digits(3)} {digits(4)}"]), "PHONE")


# Verhoeff tables (Aadhaar check digit)
_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_VINV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_check_digit(num: str) -> int:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[(i + 1) % 8][int(ch)]]
    return _VINV[c]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(num)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def aadhaar() -> P:
    base = digits(11, "23456789")
    n = base + str(verhoeff_check_digit(base))
    s = pick(["sp", "sp", "plain", "dash", "masked"])
    v = {"sp": f"{n[:4]} {n[4:8]} {n[8:]}", "plain": n, "dash": f"{n[:4]}-{n[4:8]}-{n[8:]}",
         "masked": f"XXXX XXXX {n[8:]}"}[s]
    return P(v, "ID")


def pan(p: Person) -> P:
    return P(letters(3) + "P" + p.last[0].upper() + digits(4) + letters(1), "ID")


def abha_number() -> P:
    n = "91" + digits(12)
    return P(pick([f"{n[:2]}-{n[2:6]}-{n[6:10]}-{n[10:]}", n]), "ID")


def abha_address(p: Person) -> P:
    return P(f"{p.first.split()[0].lower()}{p.last[0].lower()}{R.randint(10, 99)}@abdm", "ID")


def email(p: Person, kind: str = "patient") -> P:
    f = re.sub(r"[^a-z]", "", p.first.split()[0].lower())
    l = re.sub(r"[^a-z]", "", p.last.split()[0].lower())
    dom = pick(["example.com", "example.org", "example.net"])
    if kind == "doctor":
        v = pick([f"dr.{l}{R.randint(1, 99)}@{dom}", f"{f}.{l}.md@{dom}", f"dr{f}{l[0]}@{dom}"])
    else:
        v = pick([f"{f}.{l}{R.randint(1, 999)}@{dom}", f"{f}{l[0]}{R.randint(70, 99)}@{dom}", f"{l}.{f}@{dom}"])
    return P(v, "EMAIL")


def hosp_email(name: str) -> P:
    slug = re.sub(r"[^a-z]", "", name.lower())[:14]
    return P(f"{pick(['info', 'discharge', 'helpdesk', 'mrd', 'care'])}.{slug}@{pick(['example.com', 'example.org', 'example.net'])}", "EMAIL")


def pin(city: str) -> P:
    return P(CITIES[city][1] + digits(3), "LOCATION")


def vehicle(state: str) -> P:
    code = VEH_STATE.get(state, "TS")
    a, b, c = f"{R.randint(1, 40):02d}", letters(pick([1, 2, 2])), digits(4, "123456789")
    return P(pick([f"{code} {a} {b} {c}", f"{code}-{a}-{b}-{c}", f"{code}{a}{b}{c}", f"{code} {a} {b}{c}"]), "OTHER_PHI")


def reg_no(state: str):
    council = STATE_COUNCIL.get(state, pick(COUNCILS))
    num = digits(5, "123456789")
    s = pick(["paren", "labelled", "slash", "mci"])
    if s == "paren":
        return ["Reg. No. ", P(num, "ID"), f" ({council})"]
    if s == "labelled":
        return [f"{council} Reg. No: ", P(num, "ID")]
    if s == "slash":
        return ["Regd. No.: ", P(f"{council}/{pick(['FMR', 'R', 'REG'])}/{num}", "ID")]
    return ["Reg. No.: ", P(f"MCI-{R.randint(10, 99)}-{num}", "ID")]


def hosp_prefix(name: str) -> str:
    caps = "".join(w[0] for w in re.findall(r"[A-Za-z']+", name) if w[0].isupper())
    return (caps[:3] or "HSP").upper()


def uhid(hname: str, year: int) -> P:
    pf = hosp_prefix(hname)
    s = pick(["pfx_yy", "dot", "city", "plain", "dash"])
    yy = year % 100
    v = {"pfx_yy": f"{pf}{yy}{R.randint(1, 12):02d}{digits(5)}", "dot": f"{pf}.{digits(10, '0')}",
         "city": f"{pf}/{year}/{digits(5)}", "plain": digits(pick([8, 9, 10]), "123456789"),
         "dash": f"{pf}-{digits(6)}"}[s]
    return P(v, "ID")


def ip_no(year: int) -> P:
    yy = year % 100
    return P(pick([f"IP/{yy}/{digits(6)}", f"IPD-{year}-{digits(5)}", f"{yy}-{digits(5)}", f"IP{yy}{digits(6)}",
                   digits(7, "123456789")]), "ID")


def generic_facility(town: str) -> list:
    pat = pick(GENERIC_FACILITY)
    pre, post = pat.split("{t}")
    return [pre, P(town, "LOCATION"), post]


# ======================================================================= context (one fictional encounter)
class Ctx:
    def __init__(self, doc_type: str):
        self.doc_type = doc_type
        self.region = R.choices(list(REGION_WEIGHTS), weights=list(REGION_WEIGHTS.values()))[0]
        cities = [c for c in REGIONS[self.region]["cities"] if c in CITIES]
        self.city = pick(cities)
        self.state, _, self.std, self.locs, self.towns = CITIES[self.city]
        # patient
        if doc_type == "obstetric":
            self.sex, self.age = "F", R.randint(19, 38)
        else:
            self.sex = pick("MF")
            r = R.random()
            self.age = (R.randint(90, 101) if r < 0.09 else R.randint(1, 13) if r < 0.15 else R.randint(14, 89))
        self.pt = new_person(self.region if chance(0.85) else None, self.sex)
        self.married = self.age >= 22 and chance(0.85)
        rel_sex = "M" if (self.sex == "M" or not self.married or self.age < 18) else "M"
        self.rel = new_person(self.pt.region, rel_sex)
        if self.sex == "M" or not self.married:
            self.rel_marker = pick(["S/O", "S/O", "Son of", "s/o"]) if self.sex == "M" else pick(["D/O", "D/O", "Daughter of"])
            self.rel_word = "father"
            self.rel.last = self.pt.last if chance(0.7) else self.rel.last
        else:
            self.rel_marker = pick(["W/O", "W/O", "Wife of", "w/o"])
            self.rel_word = "husband"
            self.rel.last = self.pt.last if chance(0.6) else self.rel.last
        if self.rel.last in FEMALE_ONLY_SURNAMES:
            r = REGIONS[self.rel.region]
            self.rel.last = pick(r.get("sur_m") or r["sur"]) if not r.get("sur_f_only") else pick(r["sur"])
        while self.rel.first == self.pt.first:
            self.rel.first = pick(REGIONS[self.rel.region]["m"])
        # attendant: the relation decides the sex (no "father Mamta")
        rels = ["son", "daughter", "brother", "father", "mother", "nephew", "son-in-law", "sister"]
        rels += ["wife"] if self.sex == "M" else ["husband"]
        self.att_relation = pick(rels)
        att_sex = "F" if self.att_relation in ("daughter", "mother", "wife", "sister") else "M"
        self.attendant = new_person(self.pt.region, att_sex)
        self.religion = {"muslim": "Muslim", "christian": "Christian", "punjabi": "Sikh"}.get(self.pt.region, "Hindu")
        # hospital
        self.govt = doc_type in ("discharge_govt",) or (doc_type in ("death", "nursing") and chance(0.3))
        self.hosp_city = self.city if chance(0.7) else pick([c for c in CITIES if c in CITY_HOSP or c in GOVT_NAMED] or [self.city])
        hs, _, hstd, hlocs, htowns = CITIES[self.hosp_city]
        self.hosp_state, self.hosp_std, self.hosp_locs = hs, hstd, hlocs
        if self.govt:
            self.hosp_name = pick(GOVT_NAMED.get(self.hosp_city, [None])) if chance(0.75) else None
        else:
            pool = CITY_HOSP.get(self.hosp_city, [])
            self.hosp_name = pick(pool) if pool and chance(0.6) else f"{pick(LOCAL_HOSP_PREFIX)} {pick(LOCAL_HOSP_SUFFIX)}"
        self.generic_town = pick(htowns)
        self.generic_pat = (pick(["Government General Hospital, {t}", "District Hospital, {t}", "Civil Hospital, {t}"])
                            if doc_type in ("death", "nursing", "discharge_govt") else None)
        # dates
        self.doa = dt.date(2021, 1, 1) + dt.timedelta(days=R.randint(0, 1730))
        self.los = R.randint(1, 14)
        self.dod = self.doa + dt.timedelta(days=self.los)
        self.fu = self.dod + dt.timedelta(days=pick([5, 7, 7, 10, 14, 15, 30]))
        self.date_style = pick(DATE_STYLES)
        self.time_style = pick(TIME_STYLES)
        self.year = self.doa.year
        # clinical team
        self.dept = pick(list(DEPTS))
        self.cons = doctor()
        self.cons2 = doctor()
        self.resident = doctor()
        self.ref_doc = doctor()

    # convenience
    def d(self, date: dt.date, style: str | None = None) -> P:
        return D(date, style or self.date_style)

    def t(self, hour: int | None = None) -> P:
        return T(self.time_style if chance(0.8) else None, hour)

    def pt_name(self, header: bool = True) -> P:
        return self.pt.render(name_style(self.pt, header))

    def hosp_org(self) -> list:
        if self.hosp_name:
            return [P(self.hosp_name.upper() if chance(0.3) else self.hosp_name, "ORGANIZATION")]
        if self.generic_pat:
            pre, post = self.generic_pat.split("{t}")
            return [pre, P(self.generic_town, "LOCATION"), post]
        return generic_facility(self.generic_town)

    def street(self, locs: list[str]) -> P:
        loc = pick(locs)
        s = pick([
            f"H.No. {R.randint(1, 22)}-{R.randint(1, 12)}-{R.randint(1, 450)}/{pick('ABCD')}, {pick(STREETS)}, {loc}",
            f"Flat No. {R.randint(1, 9)}0{R.randint(1, 9)}, {pick(BUILDINGS)}, {loc}",
            f"Door No. {R.randint(1, 60)}/{R.randint(1, 30)}, {pick(STREETS)}, {loc}",
            f"#{R.randint(1, 999)}, {R.randint(1, 18)}th Cross, {loc}",
            f"{pick('ABCDE')}-{R.randint(101, 909)}, {pick(BUILDINGS)}, {loc}",
            f"Plot No. {R.randint(1, 300)}, Sector {R.randint(1, 40)}, {loc}",
            f"{R.randint(1, 250)}, {pick(STREETS)}, Near {pick(LANDMARKS)}, {loc}",
            f"Opp. {pick(LANDMARKS)}, {loc}",
        ])
        return P(s, "LOCATION")

    def address(self, rural: bool | None = None) -> list:
        rural = chance(0.4) if rural is None else rural
        if not rural:
            parts = [self.street(self.locs), ", ", P(self.city, "LOCATION")]
            if chance(0.85):
                parts += [pick([" - ", " ", ", ", "-"]), pin(self.city)]
            if chance(0.6) and self.state != self.city:
                parts += [", ", self.state]
            return parts
        v, po, t = pick(self.towns), pick(self.towns), pick(self.towns)
        style = pick(["vill_po", "village_mandal", "at_post", "paren"])
        if style == "vill_po":
            parts = ["Vill. ", P(v, "LOCATION"), ", P.O. ", P(po, "LOCATION"), ", Dist. ", P(self.city, "LOCATION")]
        elif style == "village_mandal":
            parts = ["Village ", P(v, "LOCATION"), ", Mandal ", P(t, "LOCATION"), ", District ", P(self.city, "LOCATION")]
        elif style == "at_post":
            parts = ["At/Post ", P(v, "LOCATION"), ", Tal. ", P(t, "LOCATION"), ", Dist. ", P(self.city, "LOCATION")]
        else:
            parts = [P(v, "LOCATION"), " (Village), ", P(t, "LOCATION"), " (Tehsil), ", P(self.city, "LOCATION"), " (Dist.)"]
        if chance(0.7):
            parts += [pick([" - ", ", PIN ", " PIN: "]), pin(self.city)]
        if chance(0.5) and self.state != self.city:
            parts += [", ", self.state]
        return parts

    def hosp_address(self) -> list:
        parts = [self.street(self.hosp_locs) if chance(0.8) else P(pick(self.hosp_locs), "LOCATION"),
                 ", ", P(self.hosp_city, "LOCATION")]
        if chance(0.7):
            parts += [" - ", pin(self.hosp_city)]
        if chance(0.4) and self.hosp_state != self.hosp_city:
            parts += [", ", self.hosp_state]
        return parts

    def age_sex(self) -> list:
        sx = self.sex
        if self.age >= 90:
            num = P(str(self.age), "AGE")
        elif self.age >= 1:
            num = str(self.age)
        else:
            num = None
        if num is None:
            return [f"{R.randint(1, 11)} months / {('Male' if sx == 'M' else 'Female')}"]
        st = pick(["y_slash", "yrs_slash", "years_slash", "slash", "Y_sep"])
        word = "Male" if sx == "M" else "Female"
        if st == "y_slash":
            return [num, f" Y / {sx}"]
        if st == "yrs_slash":
            return [num, f" Yrs/{sx}"]
        if st == "years_slash":
            return [num, f" years / {word}"]
        if st == "slash":
            return [num, f"/{sx}"]
        return [num, f"Y/{sx}"]

    def age_words(self) -> list:
        num = P(str(self.age), "AGE") if self.age >= 90 else str(self.age)
        return pick([[num, "-year-old"], [num, " year old"], [num, " yr old"], ["aged ", num, " years"]])


# ======================================================================= clinical case library
# Each case: (key, departments, diagnosis, complaints, history, examination, investigations, course,
# medications, advice). Strings may contain {slots} filled from the narrative context (PHI-bearing).
CASES = [
    dict(key="dengue", dept="General Medicine", dx="Dengue fever (NS1 antigen positive) with thrombocytopenia", icd="A90",
         cc="Fever with chills x 5 days, body pains, retro-orbital pain, vomiting x 2 days.",
         hist="Patient was apparently normal 5 days back when he/she developed high grade fever with chills, associated with "
              "myalgia and retro-orbital pain. No h/o bleeding manifestations, malena or haematuria. Not a k/c/o DM/HTN/BA.",
         exam="Temp 101.2 F, PR 102/min, BP 100/70 mmHg, RR 18/min, SpO2 98% on RA. Tourniquet test negative. "
              "P/A: soft, mild tenderness in Rt. hypochondrium. CVS: S1S2 +, RS: B/L NVBS, CNS: NAD.",
         inv="NS1 antigen: Positive. IgM dengue: Negative. Hb 13.8 g/dL, TLC 3,200/cumm, Platelets 38,000/cumm "
             "(1,12,000/cumm at discharge). Hct 44%. SGOT/SGPT 112/96 U/L. USG abdomen: mild GB wall oedema.",
         course="Managed conservatively with IV fluids (NS/RL), antipyretics and serial platelet monitoring. No platelet "
                "transfusion required. Patient improved symptomatically and has been afebrile for the last 48 hrs.",
         meds=["Tab. Dolo 650 mg 1-1-1 x 3 days (SOS for fever)", "Tab. Pan 40 mg 1-0-0 before food x 5 days",
               "ORS 1 sachet in 1 litre water, sip frequently"],
         advice="Avoid NSAIDs (Brufen/Combiflam). Plenty of oral fluids. Repeat CBC after 3 days."),
    dict(key="typhoid", dept="General Medicine", dx="Enteric fever (Typhoid), blood culture positive for S. typhi", icd="A01.0",
         cc="Fever x 10 days, step-ladder pattern, abdominal pain and loss of appetite.",
         hist="Fever insidious in onset, gradually progressive, associated with headache and generalised weakness. "
              "H/o consumption of outside food. No h/o cough, burning micturition or rash.",
         exam="Temp 102 F, PR 88/min (relative bradycardia), BP 110/70 mmHg. Coated tongue. Mild splenomegaly. Murphy's sign negative.",
         inv="Widal: TO 1:160, TH 1:320. Blood culture: Salmonella typhi, sensitive to ceftriaxone and azithromycin. "
             "Hb 11.6 g/dL, TLC 4,800/cumm, ESR 42 mm/hr, CRP 48 mg/L.",
         course="Started on Inj. Ceftriaxone 2 g IV BD, later switched to oral azithromycin. Fever subsided on day 6 of treatment.",
         meds=["Tab. Azithromycin 500 mg 1-0-0 x 7 days", "Tab. Paracetamol 650 mg SOS", "Cap. Becosules 1-0-0 x 15 days"],
         advice="Boiled water, home-cooked food. Hand hygiene. Complete the antibiotic course."),
    dict(key="malaria", dept="General Medicine", dx="Plasmodium vivax malaria", icd="B51.9",
         cc="Intermittent fever with chills and rigors x 4 days.",
         hist="Fever with chills and rigors, relieved with sweating, occurring on alternate days. Resides in an endemic area. "
              "No h/o altered sensorium, jaundice or decreased urine output.",
         exam="Temp 103 F, PR 110/min, BP 106/68 mmHg. Pallor +. Splenomegaly 2 cm below costal margin.",
         inv="Peripheral smear: P. vivax trophozoites seen. Rapid malaria antigen: P. vivax positive. Hb 9.8 g/dL, "
             "Platelets 96,000/cumm, S. Bilirubin 1.4 mg/dL. G6PD: normal.",
         course="Treated with Inj. Artesunate followed by oral ACT and primaquine for radical cure. Afebrile since day 3.",
         meds=["Tab. Primaquine 15 mg 1-0-0 x 14 days after food", "Tab. Paracetamol 500 mg SOS", "Tab. Ferrous sulphate 1-0-0 x 1 month"],
         advice="Use mosquito nets. Report if dark urine or yellowish discolouration of eyes."),
    dict(key="age", dept="General Medicine", dx="Acute gastroenteritis with moderate dehydration and hypokalemia", icd="A09",
         cc="Loose stools 10-12 episodes/day x 2 days, vomiting x 1 day, giddiness.",
         hist="Watery, non-blood stained stools, associated with cramping abdominal pain. H/o eating at a roadside stall. "
              "Decreased urine output since morning.",
         exam="PR 112/min, BP 90/60 mmHg, dry tongue, skin turgor decreased. P/A: soft, diffuse tenderness, BS +.",
         inv="S. Na 132 mEq/L, S. K 3.1 mEq/L, S. Creatinine 1.6 mg/dL (1.0 at discharge). Stool R/E: pus cells 8-10/hpf.",
         course="Rehydrated with IV RL/NS, potassium correction (KCl), Inj. Ondansetron and oral probiotics. Stools reduced to "
                "1-2/day, tolerating orally.",
         meds=["ORS 200 ml after every loose stool", "Cap. Sporlac 1-1-1 x 5 days", "Tab. Racecadotril 100 mg 1-1-1 x 3 days"],
         advice="Soft bland diet. Return if blood in stools or persistent vomiting."),
    dict(key="pneumonia", dept="Pulmonology", dx="Community acquired pneumonia, right lower lobe", icd="J18.9",
         cc="Cough with expectoration x 7 days, fever x 5 days, breathlessness (MMRC grade II) x 2 days.",
         hist="Cough productive of yellowish sputum. Fever moderate grade. K/c/o T2DM since {year_dx} on OHA. "
              "Non-smoker.",
         exam="Temp 100.8 F, PR 104/min, RR 26/min, SpO2 91% on RA, 96% on 2 L O2. Bronchial breath sounds and coarse "
              "crepts in right infra-scapular area.",
         inv="CXR PA: right lower zone consolidation. TLC 16,800/cumm (N 86%). CRP 96 mg/L. Sputum AFB/CBNAAT: negative. "
             "RBS 248 mg/dL. CURB-65 score 1.",
         course="Treated with Inj. Augmentin 1.2 g IV TDS, Tab. Azithromycin, nebulisation with Duolin and Budecort, "
                "O2 support, insulin for glycaemic control. Weaned off oxygen on day 4.",
         meds=["Tab. Augmentin 625 mg 1-0-1 x 5 days", "Tab. Glycomet GP 1 1-0-1", "Syp. Ascoril LS 10 ml TDS x 5 days"],
         advice="Incentive spirometry. Repeat CXR after 4 weeks."),
    dict(key="stemi", dept="Cardiology", dx="Acute inferior wall STEMI, single vessel disease (RCA), s/p primary PTCA + DES to RCA", icd="I21.1",
         cc="Retrosternal chest pain, compressive, radiating to left arm, with sweating x 3 hours.",
         hist="Sudden onset chest pain at rest. K/c/o systemic hypertension since {year_dx}. Smoker (10 pack-years).",
         exam="PR 58/min, BP 100/60 mmHg, SpO2 97%. S1S2 +, no murmur. Chest: clear.",
         inv="ECG: ST elevation in II, III, aVF with reciprocal changes. Trop I: 18.4 ng/mL. 2D Echo: inferior wall hypokinesia, "
             "LVEF 45%. CAG: SVD - RCA 95% thrombotic occlusion.",
         course="Taken up for primary PTCA on {d_proc} at {t_proc}; DES deployed to RCA by Dr. {doc_b} with TIMI III flow. "
                "Post-procedure uneventful. Mobilised and cardiac rehab counselling given.",
         meds=["Tab. Ecosprin 75 mg 0-1-0 after food (continue lifelong)", "Tab. Clopilet 75 mg 0-1-0 x 1 year",
               "Tab. Atorva 80 mg 0-0-1", "Tab. Metoprolol 25 mg 1-0-1", "Tab. Sorbitrate 5 mg SL SOS for chest pain"],
         advice="Do not stop antiplatelets without consulting a cardiologist. Stop smoking. Low salt, low fat diet."),
    dict(key="stroke", dept="Neurology", dx="Acute ischaemic stroke, left MCA territory, with right hemiparesis", icd="I63.5",
         cc="Sudden onset weakness of right upper and lower limb with slurring of speech since morning.",
         hist="Weakness noticed on waking up, last seen normal at {t_event} the previous night. K/c/o HTN, irregular on treatment. "
              "Not a k/c/o Parkinson's disease or seizures.",
         exam="BP 186/104 mmHg, GCS E4V4M6, NIHSS 8. Power right side 3/5, left 5/5. Right plantar extensor. Kernig's sign negative.",
         inv="MRI brain: acute infarct in left MCA territory (DWI restriction). Carotid Doppler: 40% stenosis left ICA. "
             "Lipid profile: LDL 148 mg/dL. HbA1c 6.1%.",
         course="Outside thrombolysis window. Managed with antiplatelets, statins, BP control, Ryle's tube feeding initially and "
                "physiotherapy. Foley's catheter removed on day 5. Power improved to 4/5.",
         meds=["Tab. Ecosprin AV 75/20 0-0-1", "Tab. Telma 40 mg 1-0-0", "Tab. Pantop 40 mg 1-0-0"],
         advice="Regular physiotherapy. Swallow assessment before solid diet. Check BP daily."),
    dict(key="diabetic_foot", dept="General Surgery", dx="Uncontrolled T2DM with left diabetic foot (Wagner grade 3), s/p debridement", icd="E11.5",
         cc="Ulcer over left foot x 3 weeks with foul-smelling discharge and fever x 4 days.",
         hist="K/c/o T2DM since {year_dx}, poor compliance. Ulcer started after a minor injury while walking barefoot.",
         exam="Temp 100 F. Ulcer 4 x 3 cm over plantar aspect of left 1st metatarsal head with slough. Dorsalis pedis pulse felt.",
         inv="RBS 342 mg/dL, HbA1c 10.4%. TLC 14,200/cumm. X-ray left foot: no osteomyelitis. Pus C/S: Pseudomonas aeruginosa.",
         course="Wound debridement done under spinal anaesthesia on {d_proc} by Dr. {doc_b}. Inj. Piptaz as per culture, "
                "regular dressings, basal-bolus insulin.",
         meds=["Inj. Human Mixtard 30/70 20 U - 0 - 14 U SC before food", "Tab. Linezolid 600 mg 1-0-1 x 7 days",
               "Daily dressing with povidone iodine"],
         advice="Offloading footwear. Foot care education given. SMBG charting."),
    dict(key="tb", dept="Pulmonology", dx="Pulmonary tuberculosis (microbiologically confirmed, rifampicin sensitive)", icd="A15.0",
         cc="Cough x 2 months, evening rise of temperature, weight loss (6 kg), haemoptysis x 2 episodes.",
         hist="H/o contact with a TB patient in the family. No previous ATT. HIV non-reactive.",
         exam="Thin built, BMI 16.2. Crepts in right infraclavicular area. SpO2 95%.",
         inv="Sputum CBNAAT: MTB detected, rifampicin resistance not detected. CXR: right upper zone cavity. LFT normal.",
         course="Started on ATT (HRZE) as per NTEP weight-band; notified on Ni-kshay (Nikshay ID: {nikshay}). Linked to "
                "DOTS centre at {dots_fac} for continuation.",
         meds=["FDC ATT (HRZE) 3 tabs 1-0-0 on empty stomach", "Tab. Pyridoxine 10 mg 1-0-0", "High protein diet"],
         advice="Cough etiquette, mask. LFT after 2 weeks. Do not miss doses."),
    dict(key="op_poisoning", dept="General Medicine", dx="Organophosphorus compound poisoning (suicidal), moderate", icd="T60.0",
         cc="Alleged h/o consumption of insecticide (chlorpyrifos) at around {t_event} on {d_onset} following a family dispute.",
         hist="Brought by relatives within 2 hrs. Vomiting x 3 episodes, excessive salivation and sweating.",
         exam="Pupils pinpoint, B/L crepts, PR 54/min, BP 100/70 mmHg, GCS 13/15. Fasciculations +.",
         inv="Serum cholinesterase: 1,840 U/L (low). ABG: mild metabolic acidosis. ECG: sinus bradycardia.",
         course="Gastric lavage done, Inj. Atropine titrated to atropinisation and Inj. PAM infusion. MLC No. {mlc} registered "
                "and intimation sent to {ps} Police Station. Psychiatry evaluation done; counselling given.",
         meds=["Tab. Pan 40 mg 1-0-0 x 7 days", "Tab. Escitalopram 10 mg 0-0-1 (as per psychiatry)"],
         advice="Keep pesticides locked away. Psychiatry follow-up. Family counselling."),
    dict(key="rta", dept="General Surgery", dx="Road traffic accident with mild head injury (right frontal contusion) and abrasions", icd="S06.3",
         cc="H/o RTA on {d_onset} at about {t_event} near {place_event} while riding a two-wheeler ({veh}); brief loss of consciousness.",
         hist="Hit by a lorry from behind. H/o one episode of vomiting. No ENT bleed, no seizures. Helmet not worn.",
         exam="GCS 14/15 at admission, 15/15 at discharge. Abrasions over right forearm and knee. Pupils B/L NSRL.",
         inv="CT brain (plain): small right frontal contusion 8 x 6 mm, no midline shift. X-ray chest and pelvis: normal.",
         course="Managed conservatively with anti-oedema measures, analgesics and wound care. MLC No. {mlc} informed to "
                "{ps} PS. Neurosurgery opinion: conservative management.",
         meds=["Tab. Levipil 500 mg 1-0-1 x 7 days", "Tab. Zerodol SP 1-0-1 x 5 days", "Mupirocin ointment for abrasions"],
         advice="Avoid driving for 4 weeks. Report if headache, vomiting or drowsiness."),
    dict(key="snake_bite", dept="General Medicine", dx="Viperine snake bite with coagulopathy", icd="T63.0",
         cc="Alleged h/o snake bite over left foot while working in the fields at {place_event} at around {t_event}.",
         hist="Local swelling and pain, bleeding from bite site. Snake identified by relatives as Russell's viper.",
         exam="Swelling up to left knee, fang marks +, bleeding from gums. BP 120/80 mmHg.",
         inv="20WBCT: >20 min (unclotted). PT/INR 2.8. S. Creatinine 1.1 mg/dL. Urine: no haematuria.",
         course="ASV 10 vials given followed by repeat doses as per protocol; 20WBCT normalised at 18 hrs. Tetanus toxoid given.",
         meds=["Tab. Augmentin 625 mg 1-0-1 x 5 days", "Tab. Paracetamol 650 mg SOS", "Limb elevation"],
         advice="Review for wound check. Report if decreased urine output."),
    dict(key="appendicitis", dept="General Surgery", dx="Acute appendicitis, s/p laparoscopic appendicectomy", icd="K35.8",
         cc="Pain abdomen in right iliac fossa x 2 days with fever and vomiting.",
         hist="Pain started around the umbilicus and shifted to RIF. Anorexia +.",
         exam="Tenderness and rebound tenderness at McBurney's point. Rovsing's sign positive.",
         inv="TLC 13,600/cumm. USG abdomen: inflamed, non-compressible appendix 9 mm. Alvarado score 8.",
         course="Underwent laparoscopic appendicectomy under GA on {d_proc} by Dr. {doc_b}. Post-op period uneventful. "
                "HPE: acute suppurative appendicitis.",
         meds=["Tab. Cefuroxime 500 mg 1-0-1 x 5 days", "Tab. Ultracet 1-0-1 x 3 days", "Tab. Pan 40 mg 1-0-0"],
         advice="Suture removal on POD 7. Avoid lifting heavy weights for 4 weeks."),
    dict(key="ckd", dept="Nephrology", dx="CKD stage 5 on maintenance haemodialysis with fluid overload", icd="N18.5",
         cc="Breathlessness x 3 days, swelling of legs, decreased urine output.",
         hist="K/c/o CKD since {year_dx}, on MHD twice weekly via left radio-cephalic AV fistula. Missed 2 sessions.",
         exam="BP 170/100 mmHg, B/L pedal oedema, basal crepts, JVP raised. AV fistula thrill +.",
         inv="Urea 168 mg/dL, Creatinine 9.8 mg/dL, K 6.1 mEq/L, Hb 8.2 g/dL. CXR: B/L pulmonary oedema.",
         course="Underwent urgent haemodialysis x 3 sessions with ultrafiltration. Hyperkalemia corrected. Erythropoietin started.",
         meds=["Tab. Nicardia retard 20 mg 1-0-1", "Inj. Erythropoietin 4000 U SC twice weekly", "Tab. Sevelamer 400 mg 1-1-1 with food"],
         advice="Fluid restriction 800 ml/day. Do not miss dialysis. Low potassium diet."),
    dict(key="hip_fracture", dept="Orthopaedics", dx="Fracture neck of left femur, s/p cemented bipolar hemiarthroplasty", icd="S72.0",
         cc="Fall at home in the bathroom, pain left hip and inability to bear weight.",
         hist="Thomas splint applied at a local hospital before referral. K/c/o HTN and hypothyroidism.",
         exam="Left lower limb shortened and externally rotated. Distal neurovascular status intact.",
         inv="X-ray pelvis: displaced intracapsular fracture neck of left femur. Hb 10.4 g/dL. 2D Echo: LVEF 55%. TSH 4.2.",
         course="Cemented bipolar hemiarthroplasty done on {d_proc} by Dr. {doc_b} under spinal anaesthesia. DVT prophylaxis "
                "with Inj. Enoxaparin. Mobilised with walker on POD 2.",
         meds=["Tab. Ultracet 1-0-1 x 5 days", "Tab. Shelcal 500 mg 1-0-1", "Tab. Thyronorm 50 mcg 1-0-0 empty stomach"],
         advice="Walker-assisted walking. Avoid squatting and cross-legged sitting. Suture removal on POD 14."),
    dict(key="copd", dept="Pulmonology", dx="Acute exacerbation of COPD with type 2 respiratory failure", icd="J44.1",
         cc="Increased breathlessness x 4 days, cough with whitish sputum, drowsiness since morning.",
         hist="Ex-smoker (bidi, 30 pack-years). K/c/o COPD on inhalers since {year_dx}. Chulha smoke exposure.",
         exam="RR 30/min, SpO2 84% on RA, B/L wheeze, use of accessory muscles. GCS 13/15.",
         inv="ABG: pH 7.28, pCO2 68 mmHg. CXR: hyperinflated lungs. 2D Echo: mild PAH.",
         course="Managed with BiPAP, nebulised bronchodilators, IV steroids and antibiotics. Weaned off BiPAP on day 3.",
         meds=["Tiotropium + Formoterol rotacaps 1 puff BD", "Tab. Deriphyllin retard 150 mg 1-0-1", "Tab. Prednisolone 20 mg tapering"],
         advice="Inhaler technique demonstrated. Avoid chulha smoke. Pneumococcal and influenza vaccines advised."),
    dict(key="hepatitis_a", dept="Paediatrics", dx="Acute viral hepatitis A", icd="B15.9", child=True,
         cc="Yellowish discolouration of eyes x 5 days, loss of appetite, vomiting and high coloured urine.",
         hist="Similar illness in a neighbour's child. Immunised till date as per NIS.",
         exam="Icterus ++, liver palpable 2 cm below costal margin, soft, non-tender. No flap.",
         inv="S. Bilirubin 6.8 mg/dL, SGPT 1,450 U/L, SGOT 1,120 U/L, PT/INR 1.2. IgM anti-HAV: Positive.",
         course="Managed supportively with IV fluids, antiemetics and a high carbohydrate diet. Appetite improved.",
         meds=["Syp. Ondansetron 2 ml SOS", "Syp. Udiliv 5 ml 1-0-1 x 2 weeks", "Glucose water / sugarcane juice"],
         advice="Hand hygiene. Repeat LFT after 2 weeks. School leave for 2 weeks."),
    dict(key="febrile_seizure", dept="Paediatrics", dx="Simple febrile seizure with acute upper respiratory infection", icd="R56.0", child=True,
         cc="Fever x 1 day, one episode of generalised tonic-clonic movements lasting 2 minutes at {t_event}.",
         hist="No post-ictal deficit. Developmental milestones normal. No family h/o epilepsy.",
         exam="Temp 102.6 F, throat congested, no meningeal signs, CNS: no focal deficit.",
         inv="CBC: TLC 11,200/cumm. RBS 96 mg/dL. S. Calcium 9.2 mg/dL.",
         course="Treated with antipyretics, tepid sponging and Syp. Augmentin. No recurrence of seizures during stay.",
         meds=["Syp. Paracetamol (250 mg/5 ml) 5 ml SOS", "Syp. Augmentin DDS 3 ml 1-0-1 x 5 days"],
         advice="Parents counselled about first aid during seizures. Recurrence risk explained."),
]


def narrative_slots(c: Ctx) -> dict:
    """PHI-bearing values that case narratives may use."""
    d_onset = c.doa - dt.timedelta(days=pick([0, 0, 1]))
    d_proc = c.doa + dt.timedelta(days=R.randint(0, max(0, c.los - 1)))
    return dict(
        d_onset=c.d(d_onset, pick([c.date_style, c.date_style, "dmy_slash"])),
        d_proc=c.d(d_proc),
        t_event=T(), t_proc=c.t(),
        doc_b=doctor().render(pick(["full", "full", "first_initial"])),
        place_event=P(pick(c.locs + c.towns), "LOCATION"),
        veh=vehicle(c.state),
        mlc=P(pick([f"{R.randint(10, 999)}/{c.year}", f"MLC/{c.year % 100}/{digits(4)}", f"{R.randint(100, 999)}/{c.year % 100}"]), "ID"),
        ps=P(pick(c.locs + c.towns), "LOCATION"),
        nikshay=P(digits(8, "123456789"), "ID"),
        dots_fac=generic_facility(pick(c.towns)),
        year_dx=P(str(c.year - R.randint(2, 15)), "DATE"),
    )


def pick_case(c: Ctx) -> dict:
    if c.age < 14:
        pool = [x for x in CASES if x.get("child")] + [x for x in CASES if x["key"] in ("dengue", "typhoid", "malaria", "age")]
    elif c.age >= 70:
        pool = [x for x in CASES if x["key"] in ("hip_fracture", "copd", "stroke", "pneumonia", "stemi", "ckd", "age")]
    else:
        pool = [x for x in CASES if not x.get("child")]
    return pick(pool)


def he_she(c: Ctx, s: str) -> str:
    return s.replace("he/she", "he" if c.sex == "M" else "she")


# ======================================================================= document templates
def field_line(d: Doc, items: list[tuple[str, list]]) -> None:
    sep = pick([": ", " : ", ":\t", ":  "])
    for i, (lab, val) in enumerate(items):
        if i:
            d.add(pick(["    ", "\t", "      ", "  |  "]))
        d.add(lab, sep, val)
    d.add("\n")


def clinical_sections(d: Doc, c: Ctx, case: dict, slots: dict, upper: bool) -> None:
    H = (lambda s: s.upper()) if upper else (lambda s: s)
    d.put(f"\n{H('Chief Complaints')}:\n" + he_she(c, case["cc"]) + "\n", **slots)
    d.put(f"\n{H('History of Present Illness')}:\n" + he_she(c, case["hist"]) + "\n", **slots)
    if chance(0.25) and c.age < 80:
        fam_age = R.randint(90, 99)
        d.add(f"\n{H('Family History')}: ", pick(["Mother", "Father", "Grandmother"]), " aged ", P(str(fam_age), "AGE"),
              " years, alive, k/c/o HTN.\n")
    d.put(f"\n{H('On Examination')}:\n" + case["exam"] + "\n", **slots)
    d.put(f"\n{H('Investigations')}:\n" + case["inv"] + "\n", **slots)
    d.put(f"\n{H('Course in the Hospital')}:\n" + case["course"] + "\n", **slots)
    if chance(0.35):
        mid = c.doa + dt.timedelta(days=R.randint(0, c.los))
        d.add(pick(["Cardiology", "Nephrology", "Endocrinology", "Neurology", "Psychiatry", "Physiotherapy"]),
              " opinion was obtained on ", c.d(mid), " (", doc_title(), doctor().render(pick(["full", "last", "first_initial"])),
              ") and suggestions followed.\n")


def meds_block(d: Doc, case: dict, upper: bool) -> None:
    d.add(f"\n{'DISCHARGE MEDICATIONS' if upper else 'Medications on Discharge'}:\n")
    for i, m in enumerate(case["meds"], 1):
        d.add(f"{i}. {m}\n")


def t_discharge_corporate(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    c.dept = case["dept"]
    slots = narrative_slots(c)
    up = chance(0.5)
    d.add(c.hosp_org(), "\n", c.hosp_address(), "\n")
    if chance(0.8):
        d.add("Ph: ", landline(c.hosp_city), pick([" | ", ", ", "  "]), "Helpline: ", tollfree() if chance(0.5) else mobile(), "\n")
    if chance(0.5):
        d.add("Email: ", hosp_email(c.hosp_name or "hospital"), "\n")
    d.add("\n", pick(["DISCHARGE SUMMARY", "Discharge Summary", "DISCHARGE SUMMARY (FINAL)"]), "\n\n")
    ttl = title(c.pt, c.age, c.married)
    field_line(d, [(pick(["Patient Name", "Name of Patient", "Pt. Name", "Name"]), [ttl, c.pt_name()]),
                   (pick(["Age/Sex", "Age / Gender", "Age/Gender"]), c.age_sex())])
    field_line(d, [(pick(["UHID", "UHID No.", "Hosp. No.", "MR No."]), [uhid(c.hosp_name or "Hospital", c.year)]),
                   (pick(["IP No.", "IPD No.", "Admission No.", "IP No"]), [ip_no(c.year)])])
    rel_lab = pick(["Father/Spouse Name", "Relation", "Guardian", "S/O / W/O / D/O"])
    field_line(d, [(rel_lab, [c.rel_marker + " " if rel_lab in ("Relation",) else "",
                              pick(["Mr. ", "Shri ", ""]), c.rel.render(pick(["full", "full", "upper"]))])])
    field_line(d, [(pick(["Address", "Residential Address", "Address of Patient"]), c.address())])
    field_line(d, [(pick(["Mobile", "Contact No.", "Mob. No.", "Phone"]), [mobile()])]
               + ([("Email", [email(c.pt)])] if chance(0.25) else []))
    dstyle = c.date_style
    field_line(d, [(pick(["Date of Admission", "DOA", "Admitted On"]), [c.d(c.doa, dstyle), pick([" ", " at ", ", "]), c.t()]),
                   (pick(["Date of Discharge", "DOD", "Discharged On"]), [c.d(c.dod, dstyle), pick([" ", " at ", ", "]), c.t()])])
    cons_name = c.cons.render(pick(["full", "full", "first_initial"]))
    field_line(d, [(pick(["Consultant", "Treating Consultant", "Under Care Of", "Primary Consultant"]),
                    [doc_title(), cons_name, ", ", pick(DEPTS[c.dept]), f" ({c.dept})"])])
    if chance(0.6):
        field_line(d, [(pick(["Referred by", "Ref. Dr.", "Referring Doctor"]),
                        [doc_title(), c.ref_doc.render(pick(["full", "last", "first_initial"])), ", "] +
                        (generic_facility(pick(c.towns)) if chance(0.5) else
                         [P(f"{pick(LOCAL_HOSP_PREFIX)} {pick(LOCAL_HOSP_SUFFIX)}", "ORGANIZATION"), ", ", P(pick(c.towns), "LOCATION")]))])
    d.add("\n", "FINAL DIAGNOSIS" if up else "Final Diagnosis", ": ", case["dx"], f" (ICD-10: {case['icd']})\n")
    clinical_sections(d, c, case, slots, up)
    if chance(0.3):
        d.add("\nOccupation: ", pick(OCCUPATIONS), pick(["", f" at "]), "")
        if chance(0.5):
            d.add("works at ", P(pick(EMPLOYERS), "ORGANIZATION"))
        d.add("\n")
    d.add("\n", "CONDITION AT DISCHARGE" if up else "Condition at Discharge", ": ",
          pick(["Stable, afebrile, tolerating oral diet.", "Hemodynamically stable. Ambulatory.", "Improved, vitals stable."]), "\n")
    meds_block(d, case, up)
    d.add("\n", "ADVICE" if up else "Advice", ": ", case["advice"], "\n")
    fu_style = pick(["date", "date", "days"])
    if fu_style == "date":
        d.add("Review in ", c.dept, " OPD on ", c.d(c.fu), " with ", doc_title(), c.cons.render("last"), ". ")
    else:
        d.add("Review after ", str((c.fu - c.dod).days), " days in ", c.dept, " OPD. ")
    d.add("For appointments call ", landline(c.hosp_city) if chance(0.5) else mobile(), ".\n")
    d.add("In case of emergency (chest pain, breathlessness, bleeding, high fever) call 108 or report to the Emergency Department.\n")
    d.add("\n", doc_title(), cons_name, "\n", pick(DEPTS[c.dept]), "\n", "Consultant ", c.dept, "\n")
    d.add(reg_no(c.hosp_state), "\n")
    if chance(0.6):
        d.add("Prepared by: ", doc_title(), c.resident.render(pick(["full", "first_initial"])), " (Resident)\n")
    if chance(0.6):
        d.add("Received by: ", c.attendant.render("full"), f" ({c.att_relation})", pick([", Mob: ", " - "]), mobile(), "\n")
    return d


def t_discharge_govt(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    c.dept = case["dept"]
    slots = narrative_slots(c)
    d.add(c.hosp_org(), (["\n"] if c.hosp_name else ["\n"]))
    if c.hosp_name and chance(0.6):
        d.add(P(c.hosp_city, "LOCATION"), "\n")
    d.add(f"Department of {c.dept}, Unit {pick(['I', 'II', 'III', 'IV', 'V'])}\n")
    d.add(pick(["DISCHARGE TICKET", "DISCHARGE SLIP", "DISCHARGE SUMMARY"]), "\n\n")
    field_line(d, [("CR No.", [P(f"{c.year}{digits(9)}", "ID")]),
                   (pick(["Adm. No.", "Admission No.", "IPD Reg. No."]),
                    [P(pick([f"ADM/{c.year}/{digits(5)}", f"{digits(6)}/{c.year % 100}", digits(6)]), "ID")]),
                   ("Ward", [pick(["Male Medical Ward", "Female Medical Ward", "MICU", "Ward 22", "Surgical Ward 4", "Paediatric Ward"])])])
    field_line(d, [("Name", [title(c.pt, c.age, c.married), c.pt.render(pick(["full", "upper"])), " ", c.rel_marker, " ",
                             c.rel.render(pick(["full", "upper"]))])])
    field_line(d, [(pick(["Age/Sex", "Age"]), c.age_sex()), ("Religion", [c.religion])])
    field_line(d, [("Address", c.address(rural=chance(0.75)))])
    if chance(0.7):
        field_line(d, [(pick(["Mobile", "Mob.", "Contact"]), [mobile()])])
    dstyle = c.date_style
    field_line(d, [("Date of Admission", [c.d(c.doa, dstyle)]), ("Date of Discharge", [c.d(c.dod, dstyle)])])
    d.add("\nReferred from: ", generic_facility(pick(c.towns)), pick([" with referral slip.", ".", " (108 ambulance)."]), "\n")
    if chance(0.4):
        d.add("Brought by ASHA worker and ", c.rel_word, ".\n")
    d.add("\nDiagnosis: ", case["dx"], "\n")
    clinical_sections(d, c, case, slots, False)
    meds_block(d, case, False)
    d.add("\nAdvice: ", case["advice"], "\n")
    d.add("Follow up in ", c.dept, " OPD (Room No. ", str(R.randint(1, 40)), ") on ", c.d(c.fu, pick([dstyle, "dmy_slash"])), ".\n")
    d.add("\nJR: ", doc_title(), c.resident.render(pick(["full", "first_initial"])),
          "    SR: ", doc_title(), c.cons2.render(pick(["full", "last"])),
          "    Unit Head: ", doc_title(prof=True), c.cons.render("full"), "\n")
    return d


def t_opd(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    c.dept = case["dept"]
    clinic = P(f"{pick(LOCAL_HOSP_PREFIX)} {pick(['Clinic', 'Polyclinic', 'Health Clinic', 'Family Clinic', 'Medical Centre'])}", "ORGANIZATION")
    d.add(clinic, "\n", doc_title(), c.cons.render("full"), ", MBBS, ", pick(DEPTS[c.dept]), "\n")
    d.add(reg_no(c.state), "  |  ", c.street(c.locs), ", ", P(c.city, "LOCATION"), "  |  Ph: ", mobile(), "\n")
    d.add("Timings: 10 am - 1 pm, 6 pm - 9 pm (Sunday holiday)\n\n")
    vdate = c.doa
    field_line(d, [("Date", [c.d(vdate)]), (pick(["OP No.", "OPD No.", "Reg. No."]), [P(pick([f"OP/{c.year % 100}/{digits(6)}", f"OPD-{digits(6)}", digits(5)]), "ID")])])
    field_line(d, [("Name", [title(c.pt, c.age, c.married), c.pt_name()]), ("Age/Sex", c.age_sex())])
    if chance(0.6):
        field_line(d, [("Mob", [mobile()]), ("Place", [P(pick(c.locs + c.towns), "LOCATION")])])
    slots = narrative_slots(c)
    d.put("\nC/o: " + he_she(c, case["cc"]) + "\n", **slots)
    d.put("O/E: " + case["exam"] + "\n", **slots)
    d.add("Dx: ", case["dx"].split(",")[0], "\n\nRx\n")
    for i, m in enumerate(case["meds"], 1):
        d.add(f"{i}) {m}\n")
    d.add("\nAdv: ", case["advice"], "\n")
    if chance(0.5):
        d.add("Inv: CBC, RBS, S. Creatinine, CUE. Get reports on next visit.\n")
    gap = pick([3, 5, 7, 10])
    d.add("Review after ", str(gap), " days / on ", c.d(vdate + dt.timedelta(days=gap), pick([c.date_style, "dmy_slash"])), ".\n")
    if chance(0.3):
        d.add("Teleconsultation link: ", P(f"https://meet.example.org/{letters(3).lower()}-{letters(4).lower()}-{letters(3).lower()}", "OTHER_PHI"), "\n")
    d.add("\n", doc_title(), c.cons.render(pick(["full", "first_initial"])), "\n")
    return d


LAB_TESTS = [
    ("Haemoglobin", "g/dL", (9.0, 16.0), "13.0 - 17.0"), ("Total Leucocyte Count", "/cumm", (3000, 16000), "4000 - 11000"),
    ("Platelet Count", "lakhs/cumm", (0.4, 4.2), "1.5 - 4.5"), ("ESR", "mm/1st hr", (5, 80), "0 - 20"),
    ("Random Blood Sugar", "mg/dL", (70, 380), "70 - 140"), ("HbA1c", "%", (4.8, 12.5), "< 5.7"),
    ("Serum Creatinine", "mg/dL", (0.5, 9.5), "0.6 - 1.2"), ("Blood Urea", "mg/dL", (12, 180), "15 - 40"),
    ("Serum Sodium", "mEq/L", (125, 146), "135 - 145"), ("Serum Potassium", "mEq/L", (2.9, 6.4), "3.5 - 5.1"),
    ("Total Bilirubin", "mg/dL", (0.3, 7.5), "0.2 - 1.2"), ("SGPT (ALT)", "U/L", (12, 1500), "< 45"),
    ("TSH", "uIU/mL", (0.2, 12.0), "0.35 - 5.5"), ("LDL Cholesterol", "mg/dL", (60, 210), "< 100"),
]


def t_lab(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    lab = pick(LABS)
    d.add(P(lab, "ORGANIZATION"), pick([" (NABL Accredited)", "", " - A NABL accredited laboratory"]), "\n")
    d.add(c.street(c.locs), ", ", P(c.city, "LOCATION"), pick([" - ", " "]), pin(c.city), "\n")
    d.add("Tel: ", landline(c.city), "   Email: ", hosp_email(lab), "\n\n")
    coll = c.doa
    field_line(d, [("Patient Name", [title(c.pt, c.age, c.married), c.pt_name()]),
                   ("Age / Gender", c.age_sex())])
    field_line(d, [("Referred By", [doc_title(), c.ref_doc.render(pick(["full", "last", "first_initial"]))] if chance(0.85) else ["SELF"]),
                   (pick(["Sample ID", "Lab No.", "SID"]), [P(f"{hosp_prefix(lab)}{coll.strftime('%y%m%d')}{digits(4)}", "ID")])])
    field_line(d, [("Registered On", [c.d(coll), " ", c.t(R.randint(7, 11))]),
                   ("Collected On", [c.d(coll), " ", c.t(R.randint(7, 12))])])
    field_line(d, [("Reported On", [c.d(coll + dt.timedelta(days=pick([0, 0, 1]))), " ", c.t(R.randint(13, 21))]),
                   (pick(["Patient ID", "UHID", "Visit No."]), [P(digits(pick([7, 8, 9]), "123456789"), "ID")])])
    d.add("\n", pick(["BIOCHEMISTRY & HAEMATOLOGY", "COMPLETE HEALTH PROFILE", "TEST REPORT"]), "\n")
    d.add("Test Name                     Result      Unit          Bio. Ref. Interval\n")
    for name, unit, (lo, hi), ref in R.sample(LAB_TESTS, R.randint(5, 9)):
        v = R.uniform(lo, hi)
        val = f"{v:.1f}" if hi < 50 else f"{int(v):,}"
        d.add(f"{name:<30}{val:<12}{unit:<14}{ref}\n")
    d.add("\nInterpretation: Please correlate clinically. Values flagged H/L are outside the reference interval.\n")
    d.add("*** End of Report ***\n\n")
    d.add(doc_title(), doctor().render("full"), ", MD (Pathology)\nConsultant Pathologist\n")
    if chance(0.5):
        d.add("View report online: ", P(f"https://reports.example.org/r/{letters(2)}{digits(6)}", "OTHER_PHI"), "\n")
    return d


def t_referral(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    c.dept = case["dept"]
    d.add("Date: ", c.d(c.doa), "\n\nTo,\n", doc_title(), c.cons.render("full"), ", ", pick(DEPTS[c.dept]), "\n")
    d.add("Department of ", c.dept, "\n", c.hosp_org(), ", ", P(c.hosp_city, "LOCATION"), "\n\n")
    d.add(pick(["Respected Sir/Madam,", "Dear Doctor,", "Respected Sir,"]), "\n\n")
    ttl = title(c.pt, c.age, c.married)
    d.add("Sub: Referral of ", ttl, c.pt.render("full"), ", ", c.age_words(), "\n\n")
    num = P(str(c.age), "AGE") if c.age >= 90 else str(c.age)
    d.add("I am referring ", ttl, c.pt.render(pick(["full", "first", "last"])), ", a ", num, pick(["-year-old", " year old"]), " ",
          pick(OCCUPATIONS), " from ", P(pick(c.towns + [c.city]), "LOCATION"), ", who presented to our clinic on ",
          c.d(c.doa - dt.timedelta(days=pick([0, 1, 2])), pick([c.date_style, "dthmonthy"])), " with complaints of: ")
    slots = narrative_slots(c)
    d.put(he_she(c, case["cc"]) + " ", **slots)
    d.put(he_she(c, case["hist"]) + " ", **slots)
    d.add("\n\nOn examination: ", case["exam"], "\nInvestigations: ", case["inv"].split(". ")[0], ".\n\n")
    d.add("Provisional diagnosis: ", case["dx"].split(",")[0], ". He/She is being referred for further evaluation and "
          "management. Kindly do the needful.\n\n".replace("He/She", "He" if c.sex == "M" else "She"))
    if chance(0.4):
        d.add("Contact of attendant (", c.att_relation, " ", c.attendant.render("first"), "): ", mobile(), "\n\n")
    d.add("Thanking you,\nYours sincerely,\n\n", doc_title(), c.ref_doc.render("full"), ", MBBS", pick(["", ", MD (Medicine)", ", DNB (Family Medicine)"]), "\n")
    d.add(P(f"{pick(LOCAL_HOSP_PREFIX)} {pick(['Clinic', 'Nursing Home', 'Hospital'])}", "ORGANIZATION"), ", ",
          P(pick(c.towns), "LOCATION"), "\nMob: ", mobile(), pick(["  |  ", "\n"]), "Email: ", email(c.ref_doc, "doctor"), "\n")
    return d


def t_insurance(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    c.dept = case["dept"]
    d.add(pick(["REQUEST FOR CASHLESS HOSPITALISATION FOR HEALTH INSURANCE POLICY (PART C)",
                "DISCHARGE SUMMARY - TPA / INSURANCE COPY", "PRE-AUTHORIZATION REQUEST FORM"]), "\n\n")
    d.add("Details of the patient\n")
    dob = dt.date(c.doa.year - c.age, R.randint(1, 12), R.randint(1, 28))
    field_line(d, [("Name of the Patient", [c.pt.render(pick(["full", "upper"]))]), ("Gender", ["Male" if c.sex == "M" else "Female"])])
    field_line(d, [("Age", [P(str(c.age), "AGE") if c.age >= 90 else str(c.age), " years"]),
                   ("Date of Birth", [D(dob, pick(["dmy_slash", "dmy_dash", "dmony_dash"]))])])
    field_line(d, [("Contact No.", [mobile()]), ("Alternate Contact No.", [mobile() if chance(0.6) else landline(c.city)])])
    field_line(d, [("Policy No.", [P(pick([f"{digits(4)}/{digits(8)}/{digits(2)}/{digits(3)}", f"P/{digits(6)}/01/{c.year}/{digits(6)}",
                                           f"OG-{c.year % 100}-{digits(4)}-{digits(4)}-{digits(8)}"]), "ID")]),
                   (pick(["Member ID", "Card ID", "TPA ID"]), [P(pick([f"{letters(3)}{digits(8)}", f"{letters(2)}/{c.year}/{digits(5)}"]), "ID")])])
    if chance(0.4):
        field_line(d, [("Employee ID", [P(pick([f"EMP{digits(5)}", f"{letters(3)}/{digits(5)}"]), "ID")]),
                       ("Employer", [P(pick(EMPLOYERS), "ORGANIZATION")])])
    field_line(d, [("Aadhaar No.", [aadhaar()]), ("PAN", [pan(c.pt)])])
    if chance(0.6):
        field_line(d, [("ABHA No.", [abha_number()]), ("ABHA Address", [abha_address(c.pt)])])
    if chance(0.3):
        field_line(d, [("PM-JAY ID", [P(letters(1) + "".join(pick("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)), "ID")])])
    field_line(d, [("Occupation", [pick(OCCUPATIONS)])])
    field_line(d, [("Address", c.address())])
    field_line(d, [("Email ID", [email(c.pt)])])
    d.add("\nDetails of the treating doctor and hospital\n")
    field_line(d, [("Name of the Treating Doctor", [doc_title(), c.cons.render("full")]), ("Contact No.", [mobile()])])
    field_line(d, [("Name of the Hospital", c.hosp_org()), ("City", [P(c.hosp_city, "LOCATION")])])
    field_line(d, [("Hospital Contact", [landline(c.hosp_city)]), ("Hospital Email", [hosp_email(c.hosp_name or "hospital")])])
    field_line(d, [("Date of Admission", [c.d(c.doa)]), ("Time of Admission", [c.t()])])
    field_line(d, [("Date of Discharge", [c.d(c.dod)]), ("Time of Discharge", [c.t()])])
    field_line(d, [("UHID / IP No.", [uhid(c.hosp_name or "Hospital", c.year), " / ", ip_no(c.year)])])
    d.put("\nNature of illness / presenting complaints: " + he_she(c, case["cc"]) + "\n", **narrative_slots(c))
    d.add("Provisional diagnosis: ", case["dx"], f"\nICD-10 code: {case['icd']}\n")
    d.add("Line of treatment: ", pick(["Medical management", "Surgical management", "Intensive care"]), "\n")
    if chance(0.5):
        d.add("Claim / Intimation No.: ", P(f"CLM{c.doa.strftime('%y%m%d')}{digits(4)}", "ID"), "\n")
    d.add("\nDeclaration by the patient / representative:\nI, ", c.attendant.render("full"), f", {c.att_relation} of the patient, "
          "agree to allow the hospital to provide all information to the insurer/TPA.\n")
    d.add("Signature: ________   Date: ", c.d(c.doa), "\n")
    return d


def t_nursing(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    d.add("PROGRESS NOTES / NURSING RECORD - ", c.hosp_org(), "\n")
    field_line(d, [("Patient", [c.pt_name()]), ("UHID", [uhid(c.hosp_name or "Hospital", c.year)]), ("Unit", [pick(["MICU", "SICU", "HDU", "Ward 7"])])])
    day = c.doa
    nurses = [new_person(c.region, "F") for _ in range(2)]
    for k in range(R.randint(2, 4)):
        d.add("\n", c.d(day), "\n")
        for hour in sorted(R.sample(range(0, 24), R.randint(2, 4))):
            d.add(c.t(hour), " - ")
            ev = pick(["vitals", "doctor", "drug", "attendant", "lab"])
            if ev == "vitals":
                d.add(f"BP {R.randint(96, 170)}/{R.randint(56, 100)} mmHg, PR {R.randint(60, 120)}/min, SpO2 {R.randint(88, 99)}%, "
                      f"temp {R.uniform(97.5, 102.5):.1f} F. GRBS {R.randint(80, 320)} mg/dl.\n")
            elif ev == "doctor":
                d.add("Seen by ", doc_title(), doctor().render(pick(["full", "last", "first_initial"])), " (", pick(list(DEPTS)),
                      "). Orders noted.\n")
            elif ev == "drug":
                d.add(pick(["Inj. Pantop 40 mg IV given.", "Inj. Ceftriaxone 1 g IV given.", "Nebulisation with Duolin given.",
                            "Inj. Lasix 40 mg IV given; urine output 600 ml.", "Inj. Paracetamol 1 g IV given for fever."]),
                      " As per order of ", doc_title(), c.resident.render(pick(["last", "full"])), ".\n")
            elif ev == "attendant":
                d.add("Condition explained to attendant (", c.att_relation, " ", c.attendant.render(pick(["full", "first"])),
                      ", Mob ", mobile(), "). Consent taken.\n")
            else:
                d.add("Blood samples sent for CBC, RFT, LFT. Report: Hb ", f"{R.uniform(8, 14):.1f}", " g/dL.\n")
        d.add("Signed: ", pick(["Sr. ", "SN ", "Staff Nurse ", "Sister "]), pick(nurses).render(pick(["full", "first"])), "\n")
        day = day + dt.timedelta(days=1)
    return d


def t_radiology(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    centre = pick(LABS + [f"{pick(LOCAL_HOSP_PREFIX)} {pick(['Scans', 'Imaging Centre', 'Diagnostics'])}"])
    d.add(P(centre, "ORGANIZATION"), "\n", c.street(c.locs), ", ", P(c.city, "LOCATION"), " | Ph: ", landline(c.city), "\n\n")
    field_line(d, [("Patient Name", [title(c.pt, c.age, c.married), c.pt_name()]), ("Age/Sex", c.age_sex())])
    field_line(d, [("Patient ID", [P(digits(8, "123456789"), "ID")]),
                   ("Accession No.", [P(pick([f"CT/{c.year % 100}/{digits(5)}", f"RAD-{c.doa.strftime('%y%m')}-{digits(5)}", f"MR{digits(7)}"]), "ID")])])
    field_line(d, [("Ref. by", [doc_title(), c.ref_doc.render(pick(["full", "last"]))]), ("Date", [c.d(c.doa)])])
    study = pick([
        ("CT BRAIN (PLAIN)", "Small hypodense area measuring 12 x 8 mm noted in the left gangliocapsular region. Ventricles normal "
         "in size. No midline shift. Calvarium intact.", "Acute lacunar infarct in left gangliocapsular region."),
        ("USG ABDOMEN & PELVIS", "Liver 15.2 cm, mildly enlarged with increased echotexture. GB: multiple calculi, largest 9 mm. "
         "CBD 4 mm. Both kidneys normal in size; right kidney 10.2 x 4.8 cm. No free fluid.", "Grade I fatty liver. Cholelithiasis."),
        ("MRI LUMBOSACRAL SPINE", "Diffuse disc bulge at L4-L5 with posterocentral protrusion indenting the thecal sac and "
         "compressing bilateral traversing L5 nerve roots. L5-S1: mild disc desiccation.", "L4-L5 disc protrusion with B/L L5 root compression."),
        ("CHEST X-RAY PA VIEW", "Inhomogeneous opacity in right lower zone. Costophrenic angles clear. Cardiac size normal (CTR 0.48).",
         "Right lower zone consolidation - ? pneumonia. Clinical correlation."),
        ("HRCT CHEST", "Ground glass opacities in bilateral lower lobes, CT severity score 8/25. No pleural effusion.",
         "Bilateral GGOs - infective aetiology likely."),
    ])
    d.add("\n", study[0], "\n\nFINDINGS:\n", study[1], "\n\nIMPRESSION:\n", study[2], "\n\n")
    if chance(0.3):
        d.add("Previous study dated ", c.d(c.doa - dt.timedelta(days=R.randint(60, 700))), " reviewed for comparison.\n")
    d.add(doc_title(), doctor().render("full"), ", ", pick(["MD (Radio-diagnosis)", "DMRD, DNB", "MD (Radiology)"]),
          "\nConsultant Radiologist\n")
    d.add("This report is not valid for medico-legal purposes.\n")
    return d


def t_obstetric(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    c.dept = "Obstetrics & Gynaecology"
    d.add(c.hosp_org(), "\n", c.hosp_address(), "\n\nDISCHARGE SUMMARY - OBSTETRICS\n\n")
    mother = c.pt
    field_line(d, [("Name", [pick(["Smt. ", "Mrs. ", ""]), c.pt_name()]), ("W/O", [c.rel.render(pick(["full", "upper"]))])])
    field_line(d, [("Age", [str(c.age), " yrs"]), ("IP No.", [ip_no(c.year)]), ("UHID", [uhid(c.hosp_name or "Hospital", c.year)])])
    field_line(d, [("Address", c.address())])
    field_line(d, [("Mobile", [mobile()])])
    lmp = c.doa - dt.timedelta(days=R.randint(255, 285))
    edd = lmp + dt.timedelta(days=280)
    score = pick(["G2P1L1", "G1P0", "G3P2L2", "G2P1L1A0"])
    field_line(d, [("LMP", [c.d(lmp)]), ("EDD", [c.d(edd)]), ("Obstetric score", [score])])
    field_line(d, [("Date of Admission", [c.d(c.doa), " ", c.t()]), ("Date of Discharge", [c.d(c.dod)])])
    birth = c.doa + dt.timedelta(days=pick([0, 0, 1]))
    mode = pick(["Lower segment caesarean section (LSCS)", "Normal vaginal delivery", "Assisted vaginal delivery (vacuum)"])
    d.add("\nDiagnosis: ", score, " at ", str(R.randint(37, 40)), " weeks of gestation, ", pick(["previous LSCS", "PROM", "fetal distress", "post-dated"]), "\n")
    d.add("Delivered on ", c.d(birth), " at ", T(), " by ", mode, " under ", pick(["spinal anaesthesia", "no anaesthesia"]), ".\n")
    d.add("Baby details: B/O ", mother.render(pick(["first", "full"])), f", {pick(['Male', 'Female'])}, birth weight "
          f"{R.uniform(2.3, 3.6):.2f} kg, APGAR 8/10 and 9/10. Baby UHID: ", uhid(c.hosp_name or "Hospital", c.year), "\n")
    d.add("Vaccination: BCG, OPV-0 and Hepatitis B given on ", c.d(birth + dt.timedelta(days=1)), ".\n")
    d.add("Course: Post-natal period uneventful. Breastfeeding established. Suture line healthy.\n")
    d.add("\nMedications on discharge:\n1. Tab. Ferrous ascorbate 100 mg 0-1-0 x 3 months\n2. Tab. Calcium 500 mg 1-0-1\n"
          "3. Tab. Cefuroxime 500 mg 1-0-1 x 5 days\n")
    d.add("\nAdvice: Exclusive breastfeeding. Follow up on ", c.d(c.fu), " for suture check. Immunisation as per schedule.\n")
    d.add("\nObstetrician: ", doc_title(), c.cons.render("full"), ", ", pick(DEPTS[c.dept]), "\nPaediatrician: ", doc_title(),
          c.cons2.render(pick(["full", "last"])), ", ", pick(DEPTS["Paediatrics"]), "\n")
    return d


def t_death(c: Ctx) -> Doc:
    d = Doc(c.doc_type)
    case = pick_case(c)
    d.add(c.hosp_org(), "\n\nDEATH SUMMARY\n\n")
    field_line(d, [("Name", [title(c.pt, c.age, c.married), c.pt_name()]), ("Age/Sex", c.age_sex())])
    field_line(d, [("UHID", [uhid(c.hosp_name or "Hospital", c.year)]), ("IP No.", [ip_no(c.year)])])
    field_line(d, [("Address", c.address())])
    field_line(d, [("Date of Admission", [c.d(c.doa), " ", c.t()])])
    field_line(d, [("Date and Time of Death", [c.d(c.dod), pick([" at ", " "]), c.t()])])
    d.put("\nPresenting complaints: " + he_she(c, case["cc"]) + "\n", **narrative_slots(c))
    d.add("Course: Despite maximal medical management including ventilatory and inotropic support, the patient developed "
          "refractory shock and multi-organ dysfunction. CPR was given as per ACLS protocol for 30 minutes but could not be revived.\n")
    d.add("Cause of death: (a) Septic shock (b) ", case["dx"].split(",")[0], "\n")
    d.add("\nBody handed over to ", c.att_relation, " ", c.attendant.render("full"), " (Aadhaar: ", aadhaar(), ") at ", c.t(), ".\n")
    if chance(0.5):
        d.add("Intimation sent to ", P(pick(c.locs + c.towns), "LOCATION"), " Police Station vide MLC No. ",
              P(f"{R.randint(10, 999)}/{c.year}", "ID"), ".\n")
    d.add("\n", doc_title(), c.cons.render("full"), ", ", pick(DEPTS["General Medicine"]), "\n", reg_no(c.hosp_state), "\n")
    return d


TEMPLATES = {
    "discharge_corporate": (t_discharge_corporate, 25), "discharge_govt": (t_discharge_govt, 15), "opd": (t_opd, 12),
    "lab": (t_lab, 10), "referral": (t_referral, 8), "insurance": (t_insurance, 10), "nursing": (t_nursing, 8),
    "radiology": (t_radiology, 6), "obstetric": (t_obstetric, 4), "death": (t_death, 2),
}


# ======================================================================= quality checks
def pool_words() -> set[str]:
    words = set()
    for r in REGIONS.values():
        for k in ("m", "f", "sur", "sur_f", "sur_m", "sur_f_only"):
            for v in r.get(k, []):
                words.update(w for w in re.split(r"[\s.]+", v) if len(w) > 2)
    for city, (_, _, _, locs, towns) in CITIES.items():
        words.add(city)
        words.update(locs)
        words.update(towns)
    for lst in list(CITY_HOSP.values()) + list(GOVT_NAMED.values()):
        words.update(lst)
    words.update(LABS)
    words.update(EMPLOYERS)
    # words that are also ordinary clinical / English words are not usable for the scan
    words -= {"Hope", "Noble", "Lotus", "Surya", "Mirza", "Model Town", "Civil Lines", "Old Town", "Urban Estate",
              "Raja Park", "Gokul Road", "Mavoor Road", "Kolar Road", "Raiya Road", "Kalawad Road", "Lawrence Road",
              "Station Road", "Gangapur Road", "Boring Road", "Zoo Road"}
    return words


INDEPENDENT_PATTERNS = {
    "mobile10": re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
    "plus91": re.compile(r"\+91[\s-]?\d"),
    "grouped12": re.compile(r"(?<!\d)\d{4}[ -]\d{4}[ -]\d{4}(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "url": re.compile(r"https?://\S+"),
    "num_date": re.compile(r"(?<![\d/.-])\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})(?![\d/.-])"),
}


def qa_note(note: dict, words: set[str]) -> list[str]:
    text, spans = note["text"], note["spans"]
    issues = []
    cov = [(s["start"], s["end"]) for s in spans]
    for a, b in zip(spans, spans[1:]):
        if a["end"] > b["start"]:
            issues.append("overlap")
    for s in spans:
        v = text[s["start"]:s["end"]]
        if not v or v != v.strip():
            issues.append(f"edge_whitespace:{s['label']}")
    if re.search(r"\{[a-z_0-9]+\}", text):
        issues.append("unfilled_template_slot")
    neg = [(m.start(), m.end()) for ph in HARD_NEGATIVE_PHRASES for m in re.finditer(re.escape(ph), text)]

    def covered(a, b):
        return any(s <= a and b <= e for s, e in cov) or any(s <= a and b <= e for s, e in neg)

    for w in words:
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", text):
            if not covered(m.start(), m.end()):
                issues.append(f"unannotated_pool_word:{w}")
    for name, rx in INDEPENDENT_PATTERNS.items():
        for m in rx.finditer(text):
            if not covered(m.start(), m.end()):
                issues.append(f"unannotated_pattern:{name}:{m.group()}")
    for s in spans:
        v = text[s["start"]:s["end"]]
        if s["label"] == "ID" and re.fullmatch(r"\d{4}[ -]?\d{4}[ -]?\d{4}", v):
            if not verhoeff_ok(re.sub(r"\D", "", v)) and not v.startswith("91"):
                pass  # 12-digit numbers that are not Aadhaar (e.g. card IDs) are allowed
        if s["label"] == "AGE" and int(v) < 90:
            issues.append("age_under_90_annotated")
    for m in re.finditer(r"(?<![\w-])(\d{1,3})(?=(?: ?Y(?:rs)?\b| years?\b|-year-old| yr old|/[MF]\b|Y/[MF]))", text):
        if int(m.group(1)) >= 90 and not covered(m.start(1), m.end(1)):
            issues.append("age_90_plus_not_annotated")
    return issues


def aadhaar_check(note: dict) -> int:
    bad = 0
    text = note["text"]
    for m in re.finditer(r"Aadhaar(?: No\.)?: ([\dX][\dX ]{10,13}\d)", text):
        v = m.group(1)
        if "X" not in v and not verhoeff_ok(v.replace(" ", "").replace("-", "")):
            bad += 1
    return bad


# ======================================================================= main
def build(split: str, n: int, seed: int) -> list[dict]:
    R.seed(seed)
    names, weights = list(TEMPLATES), [w for _, w in TEMPLATES.values()]
    notes = []
    for i in range(n):
        doc_type = R.choices(names, weights=weights)[0]
        c = Ctx(doc_type)
        d = TEMPLATES[doc_type][0](c)
        notes.append({"note_id": f"IN-{split.upper()}-{i + 1:04d}", "split": split, "doc_type": doc_type,
                      "region": c.region, "text": d.text, "spans": d.spans})
    return notes


def render_for_review(note: dict) -> str:
    out, pos = [], 0
    t = note["text"]
    for s in note["spans"]:
        out.append(t[pos:s["start"]])
        out.append(f"[[{t[s['start']:s['end']]}|{s['label']}]]")
        pos = s["end"]
    out.append(t[pos:])
    return "".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic_in")
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--dev2", type=int, default=200, help="second development split (Experiment 14), seed 303")
    args = ap.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    words = pool_words()
    qa = {"issues": {}, "stats": {}}
    review = ["# Synthetic Indian notes - rendered for manual review", "",
              "PHI spans are shown as `[[text|LABEL]]`. Everything here is fictional.", ""]
    for split, n, seed in (("dev", args.dev, 101), ("test", args.test, 202), ("dev2", args.dev2, 303)):
        notes = build(split, n, seed)
        issues = Counter()
        examples = {}
        for note in notes:
            for iss in qa_note(note, words):
                issues[iss.split(":")[0] + (":" + iss.split(":")[1] if iss.count(":") else "")] += 1
                examples.setdefault(iss, note["note_id"])
            issues["aadhaar_bad_checksum"] += aadhaar_check(note)
        labels = Counter(s["label"] for nt in notes for s in nt["spans"])
        qa["issues"][split] = {k: v for k, v in issues.items() if v}
        qa["stats"][split] = {"notes": len(notes), "spans": sum(labels.values()), "by_label": dict(labels.most_common()),
                              "by_doc_type": dict(Counter(nt["doc_type"] for nt in notes).most_common()),
                              "by_region": dict(Counter(nt["region"] for nt in notes).most_common()),
                              "mean_chars": round(sum(len(nt["text"]) for nt in notes) / len(notes))}
        qa.setdefault("issue_examples", {})[split] = dict(list(examples.items())[:40])
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for nt in notes:
                f.write(json.dumps(nt, ensure_ascii=False) + "\n")
        if split == "dev":
            seen = Counter()
            for nt in notes:
                if seen[nt["doc_type"]] < 2:
                    seen[nt["doc_type"]] += 1
                    review += [f"## {nt['note_id']} ({nt['doc_type']}, {nt['region']})", "", "```", render_for_review(nt), "```", ""]
    (out / "qa_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (out / "samples_for_review.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps({k: qa[k] for k in ("issues", "stats")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
