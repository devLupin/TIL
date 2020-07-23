#include <iostream>
#include <deque>
using namespace std;

bool chk(int num)
{
	while (num) {
		int temp = num % 10;
		if (!(temp == 4 || temp == 7))
			return false;
		num /= 10;
	}
	return true;
}

int main(void)
{
	ios::sync_with_stdio(0);
	cin.tie(0); cout.tie(0);
	
	int n; cin >> n;
	for (int i = n; n >= 0; i--) {
		if (chk(i)) {
			cout << i;
			break;
		}
	}
}