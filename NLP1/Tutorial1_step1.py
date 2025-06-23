import glob  # 파일 경로 패턴 매칭을 위한 모듈


# 파일 경로에서 지정된 패턴에 맞는 파일 목록을 반환하는 함수
def findFiles(path):
    return glob.glob(path)  # 주어진 경로 패턴에 일치하는 모든 파일 경로를 리스트로 반환

if __name__ == "__main__":
    # 'data/names/' 디렉토리에서 확장자가 '.txt'인 모든 파일을 찾고 출력
    print(findFiles('../data/names/*.txt'))