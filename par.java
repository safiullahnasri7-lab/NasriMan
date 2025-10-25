class C{
    int x,y;
    C(int a, int b){
        x = a;
        y = b;
      
    }

    void show(){
        System.out.println(x+y);

    }
}


public class par{
    public static void main(String[] args) {
        
        C obj = new C(12,23);
        obj.show();
        

    }

}