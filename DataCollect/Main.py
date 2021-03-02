import csv
import json
import pathlib
import re
import string
import time
from pyhdfs import HdfsClient
import moment
import pudb
from termcolor import colored
from tweepy import OAuthHandler, RateLimitError, Stream
from tweepy.streaming import StreamListener


csv_head = ["text", "location"]
csv_filename = "data.csv"

APIkey = "JsLfvl8gYTbAjozeRvBP5SL9t"  # API key
APIsecret = "PnLhX1F5Lnv4jyMm2bWxjp6jIleUDEKuZVumjqsjer924ZURrO"  # API scret
atoken = "743234403669684227-9d6jzlupZ9k0DfVONXjdswMsJ0sieTq"  # access token
asecret = "wbgftRYRCFnckpTl1jOZjFOKAZLVtheiSbSPeJAEbP4tC"  # access secret

location_scope = [-124.7625, 24.5210, -66.9326, 49.3845]
languages_limits = ["en"]

def update_csv():
    local = '/Users/constantine/PycharmProjects/test02/data.csv'
    tmpLocal = '/Users/constantine/PycharmProjects/test02/tmpdata.csv'
    remote = '/data/data.csv'
    host = '127.0.0.1:9870'
    user_name = 'host'
    client = HdfsClient(hosts=host,user_name=user_name)
    if client.exists(remote):
        client.copy_to_local(remote,tmpLocal)
        client.delete(remote)
        fRead = open(local,'r')
        fWrite = open(tmpLocal,'w')
        lines = fRead.readlines()

        for line in lines:
            fWrite.writelines(lines)
        fRead.close()
        fWrite.close()
        fRead = open(local, 'r')
        lines = fRead.read()
        fRead.close()
        fWrite = open(tmpLocal, 'w')
        lines = '\n'.join(list(set(lines.split('\n')))[1:])
        fWrite.write(lines)
        fWrite.close()
        client.copy_from_local(tmpLocal,remote)


    else:
        client.copy_from_local(local, remote)




usa_reg = re.compile(
    "USA|, AL|, AK|, AZ|, AR|, CA|, CO|, CT|, DE|, FL|, GA|, HI|, ID|, IL|, IN|, IA|, KS|, KY|, LA|, ME|, MD|, MA|, MI|, MN|, MS|, MO|, MT|, NE|, NV|, NH|, NJ|, NM|, NY|, NC|, ND|, OH|, OK|, OR|, PA|, RI|, SC|, SD|, TN|, TX|, UT|, VT|, VA|, WA|, WV|, WI|, WY"
)
usa_full_name_reg = re.compile(
    "AMERICAN|UNITED STATES|ALABAMA|ALASKA|ARIZONA|ARKANSAS|CALIFORNIA|COLORADO|CONNECTICUT|DELAWARE|FLORIDA|GEORGIA|HAWAII|IDAHO|ILLINOIS|INDIANA|IOWA|KANSAS|KENTUCKY|LOUSIANA|MAINE|MARYLAND|MASSACHUSETTS|MICHIGAN|MINNESOTA|MISSISSIPPI|MISSOURI|MONTANA|NEBRASKA|NEVADA|NEW HAMPSHIRE|NEW JERSEY|NEW MEXICO|NEW YORK|NORTH CAROLINA|NORTH DAKOTA|OHIO|OKLAHOMA|OREGON|PENNSYLVANIA|RHODE ISLAND|SOUTH CAROLINA|SOUTH DAKOTA|TENNESSEE|TEXAS|UTAH|VERMONT|VIRGINIA|WASHINGTON|WEST VIRGINIA|WISCONSIN|WYOMING|NEW YORK|LOS ANGELES|CHICAGO|HOUSTON|PHOENIX|PHILADELPHIA|SAN ANTONIO|SAN DIEGO|DALLAS|SAN JOSE|AUSTIN|JACKSONVILLE|SAN FRANCISCO|COLUMBUS|INDIANAPOLIS|FORT WORTH|CHARLOTTE|SEATTLE|DENVER|EL PASO|WASHINGTON|BOSTON|DETROIT|NASHVILLE|MEMPHIS|PORTLAND|OKLAHOMA CITY|LAS VEGAS|LOUISVILLE|BALTIMORE|MILWAUKEE|ALBUQUERQUE|TUCSON|FRESNO|SACRAMENTO|MESA|KANSAS CITY|ATLANTA|LONG BEACH|COLORADO SPRINGS|RALEIGH|MIAMI|VIRGINIA BEACH|OMAHA|OAKLAND|MINNEAPOLIS|TULSA|ARLINGTON|NEW ORLEANS|WICHITA|CLEVELAND|TAMPA|BAKERSFIELD|AURORA|HONOLULU|ANAHEIM|SANTA ANA|CORPUS CHRISTI|RIVERSIDE|LEXINGTON|ST. LOUIS|STOCKTON|PITTSBURGH|ST. PAUL|CINCINNATI|ANCHORAGE|HENDERSON|GREENSBORO|PLANO|NEWARK|LINCOLN|TOLEDO|ORLANDO|CHULA VISTA|IRVINE|FORT WAYNE|JERSEY CITY|DURHAM|ST. PETERSBURG|LAREDO|BUFFALO|MADISON|LUBBOCK|CHANDLER|SCOTTSDALE|GLENDALE|RENO|NORFOLK|WINSTON-SALEM|NORTH LAS VEGAS|IRVING|CHESAPEAKE|GILBERT|GARLAND|FREMONT|BATON ROUGE|SHREVEPORT|RICHMOND|BOISE|ARLINGTON|SAN BERNARDINO|SPOKANE|DES MOINES|MODESTO|BIRMINGHAM|TACOMA|FONTANA|ROCHESTER|OXNARD|MORENO VALLEY|FAYETTEVILLE|AURORA|GLENDALE|YONKERS|HUNTINGTON BEACH|MONTGOMERY|GREAT RAPIDS|MOBILE|AUGUSTA|COLUMBUS|FORT LAUDERDALE|AMARILLO|LITTLE ROCK|KNOXVILLE|SALT LAKE CITY|NEWPORT NEWS|JACKSON|WORCESTER|PROVIDENCE|ONTARIO|BROWNSVILLE|RANCHO CUCAMONGA|TEMPE|ROCKFORD|HUNTSVILLE|SANTA CLARITA|OVERLAND PARK|GARDEN GROVE|OCEANSIDE|TALLAHASSEE|VANCOUVER|DAYTON|CHATTANOOGA|POMONA|SANTA ROSA|GRAND PRAIRIE|SALEM|CAPE CORAL|SPRINGFIELD|SPRINGFIELD|CORONA|PEMBROKE PINES|PATERSON|EUGENE|HOLLYWOOD|SALINAS|HAMPTON|PASADENA|PASADENA|PORT ST. LUCIE|KANSAS CITY|NAPERVILLE|JOLIET|SIOUX FALLS|TORRANCE|PEORIA|LANCASTER|SYRACUSE|HAYWARD|LAKEWOOD|PALMDALE|ALEXANDRIA|BRIDGEPORT|ORANGE|WARREN|ESCONDIDO|FULLERTON|MESQUITE|SUNNYVALE|CORAL SPRINGS|FORT COLLINS|ELK GROVE|STERLING HEIGHTS|SAVANNAH|MCALLEN|ELIZABETH|HARTFORD|CEDAR RAPIDS|THOUSAND OAKS|NEW HAVEN|EL MONTE|CONCORD|TOPEKA|CARROLLTON|WACO|SIMI VALLEY|WEST VALLEY|COLUMBIA|STAMFORD|BELLEVUE|FLINT|VALLEJO|SPRINGFIELD|EVANSVILLE|INGLEWOOD|ABILENE|OLATHE|LANSING|LAFAYETTE|PROVO|VISALIA|ANN ARBOR|CLARKSVILLE|PEORIA|CARY|ATHENS|BEAUMONT|COSTA MESA|DENTON|MANCHESTER|INDEPENDENCE|DOWNEY|THORNTON|GAINESVILLE|SANTA CLARA|MIRAMAR|CHARLESTON|WEST COVINA|CLEARWATER|MCKINNEY|ALLENTOWN|WATERBURY|ROSEVILLE|WESTMINSTER|NORWALK|SOUTH BEND|FAIRFIELD|ARVADA|POMPANO BEACH|BURBANK|SAN BUENAVENTURA|PUEBLO|LOWELL|NORMAN|RICHMOND|MIDLAND|ERIE|KILLEEN|ELGIN|BERKELEY|PORTSMOUTH|CAMBRIDGE|DALY CITY|ANTIOCH|GREEN BAY|WEST JORDAN|BILLINGS"
)
normal_words = re.compile("[a-zA-Z0-9 ]")
punctuation = set(string.punctuation)

sleep_times = 900
successed_nums = 0
stack_num = 10
stack = []


def success(data):
    return colored(data, "green")


def error(data):
    return colored(data, "red")


def info(data):
    return colored(data, "blue")


def remove_emoji(text):
    return "".join([i for i in text if i in punctuation or normal_words.findall(i)])


def text_cleanup(text):
    return remove_emoji(
        " ".join([i for i in text.split() if not ("@" in i or i.startswith("http"))])
    )


def clean_emoji(location):
    return "".join([i for i in location if i in punctuation or normal_words.findall(i)])


class SListener(StreamListener):
    def on_data(self, data):
        try:
            global successed_nums
            global stack

            datas = json.loads(data)

            location = datas.get("user", {}).get("location", " ")
            text = text_cleanup(datas.get("text", " "))

            is_need = False
            if usa_reg.findall(location) or usa_full_name_reg.findall(location.upper()):
                is_need = True
                package = dict(zip(csv_head, [text, location]))
                if len(stack) >= stack_num:
                    print(success(f"saved {stack_num} items"))
                    with pathlib.Path(csv_filename).open("a") as file:
                        csv_writer = csv.DictWriter(file, csv_head)

                        csv_writer.writerows(stack)
                    stack = [package]
                else:
                    stack.append(package)

                successed_nums += 1
            print(
                f"{successed_nums}【{(success if is_need else error)(location)}】: {success(text)}"
            )

            return True
        except RateLimitError:
            print(
                colored(
                    f"[RateLimitError]: after {moment.now().add(seconds=sleep_times).format('YYYY-MM-DD HH:mm:ss')}",
                    "red",
                )
            )
            time.sleep(sleep_times)
        except Exception as e:
            print(error(f"【error】: {e}"))
            return True

    def on_status(self, status):
        if status.retweeted:
            return True


if __name__ == "__main__":


    try:
        auth = OAuthHandler(APIkey, APIsecret)
        auth.set_access_token(atoken, asecret)


        myListener = SListener()
        twitterStream = Stream(auth, myListener)
        twitterStream.filter(locations=location_scope, languages=languages_limits)
        update_csv()
        print('update done')
    except KeyboardInterrupt:
        update_csv()
        print('update except done')
        pass

