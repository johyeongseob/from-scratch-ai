import unicodedata
import string


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 유니코드 문자열을 ASCII로 변환, https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(string):
    # 1. 문자열을 유니코드 정규화
    normalized = unicodedata.normalize('NFD', string)

    # 만약 문자가 허용된 범위(all_letters)에 있고, 발음 구별 기호(Mn)가 아니면 추가
    ascii_characters = []
    for character in normalized:
        if (character in all_letters) and (unicodedata.category(character) != 'Mn'):
            ascii_characters.append(character)

    # 4. 리스트를 문자열로 변환 후 반환
    return ''.join(ascii_characters)


# 테스트
if __name__ == "__main__":
    print(f"\nall_letters: {all_letters}")
    print(f"n_letters: {n_letters}")
    print(f"unicodeToAscii('Ślusàrski'): {unicodeToAscii('Ślusàrski')}")