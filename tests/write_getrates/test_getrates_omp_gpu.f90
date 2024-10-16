program test_gpu
    implicit none

    integer,parameter :: nspec = 9
    integer,parameter :: veclen = 10
    real(kind=8), dimension(veclen,nspec) :: Y
    real(kind=8), dimension(veclen) :: P, T
    real(kind=8), dimension(veclen,nspec) :: wdot_cpu, wdot_gpu
    integer :: i, j

    ! Initialize Y with random numbers
    call random_seed()
    do i = 1, veclen
        do j = 1, nspec
            call random_number(Y(i,j))
        end do
        Y(i,:) = Y(i,:)/sum(Y(i,:))
    end do

    ! Initialize P and T (you may want to set these to specific values)
    P = 101325.0d0*10.0d0 !dyn/cm^2
    T = 1200 !K

    ! Initialize wdot arrays to zero
    wdot_cpu = 0.0d0
    wdot_gpu = 0.0d0
    
    call getrates(veclen,T,Y,P,wdot_cpu)
!$omp target enter data map(to:Y,P,T,wdot_gpu)
    call getrates_gpu(veclen,T,Y,P,wdot_gpu)
!$omp target exit data map(from:wdot_gpu)

    ! Print the maximum error
    write(*,*) "Maximum error between CPU and GPU results:", maxval(abs(wdot_cpu-wdot_gpu))/maxval(abs(wdot_cpu))

end program