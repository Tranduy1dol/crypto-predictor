from binance.spot import Spot as Client

"""
Exchange functions
@param public_key: public key
@param private_key: private key
@return: client
@rtype: Client
@example: check_login(public_key, private_key)
@example: balance_of(client, "BTC")
@example: check_order(client, "BTC", "BUY")
@example: take_buy_order(client, "BTC", "BUY", 10)
"""
def check_login(public_key, private_key):
    client = Client(public_key, private_key, base_url="https://testnet.binance.vision")
    try:
        client.account(self=client)
        return client
    except:
        return False


def balance_of(client, symbol):
    account = Client.account(self=client)
    balances = account['balances']

    for balance in balances:
        if balance['asset'] == symbol:
            return (float)(balance['free'])


def check_order(client, symbol, side):
    check = False
    order = client.get_orders(symbol=symbol, limit=1)

    if order:
        last_order = order[0]
        if last_order['status'] == 'FILLED':
            check = True
            return check

        else:
            check = False
            return check


def take_buy_order(client, symbol, side, quantity):
    symbol = f"{symbol}USDT"

    order = client.new_order(symbol=symbol, quantity=quantity, side=side, type='MARKET')


def take_sell_order(client, symbol, side, quantity):
    symbol = f"{symbol}USDT"

    order = client.new_order(symbol="BTCUSDT", quantity=quantity, side=side, type='MARKET')
