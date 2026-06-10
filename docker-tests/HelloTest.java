// Compilar: javac docker-tests/HelloTest.java -d docker-tests/
// Executar: java -cp docker-tests HelloTest
public class HelloTest {

    public static void main(String[] args) {
        String versao = System.getProperty("java.version");
        System.out.println("Java version: " + versao);

        // String.strip(), repeat() e isBlank() são métodos do Java 11
        String texto = "  Olá, Docker!  ";
        assert texto.strip().equals("Olá, Docker!") : "strip() falhou";
        assert "ab".repeat(3).equals("ababab")       : "repeat() falhou";
        assert "   ".isBlank()                        : "isBlank() falhou";

        // var para inferência de tipo (Java 10+)
        var soma = somar(7, 8);
        assert soma == 15 : "somar() falhou";

        System.out.println("Todos os testes passaram!");
    }

    static int somar(int a, int b) {
        return a + b;
    }
}
